import os, glob
import json
import cv2
import numpy as np
import enum, copy
from scipy.spatial.transform import Rotation as R
os.environ["RUST_LOG"] = "error"
import glog as log

from diffusion_policy.common.lerobot_v3_io import LeRobotV3Dataset

class ObservationType(enum.Enum):
    JOINT_POSITION_ONLY = "jonit_position"
    END_EFFECTOR_POSE = "ee_pose"
    DELTA_END_EFFECTOR_POSE = "delta_ee_pose"
    JOINT_POSITION_END_EFFECTOR = "jonit_position_ee_pose"
    MASK = "mask"
    FT_ONLY = "ft_only"

class ActionType(enum.Enum):
    JOINT_POSITION = 0
    JOINT_POSITION_DELTA = 1
    END_EFFECTOR_POSE = 2
    END_EFFECTOR_POSE_DELTA = 3
    JOINT_TORQUE = 4
    COMMAND_JOINT_POSITION = 5
    COMMAND_END_EFFECTOR_POSE = 6

Action_Type_Mapping_Dict = {
    "joint_position": ActionType.JOINT_POSITION,
    "joint_position_delta": ActionType.JOINT_POSITION_DELTA,
    "end_effector_pose": ActionType.END_EFFECTOR_POSE,
    "end_effector_pose_delta": ActionType.END_EFFECTOR_POSE_DELTA,
    "command_joint_position": ActionType.COMMAND_JOINT_POSITION,
    "command_end_effector_pose": ActionType.COMMAND_END_EFFECTOR_POSE
}

class RerunEpisodeReader:
    def __init__(self, task_dir = ".", json_file="data.json", 
                 action_type: ActionType = ActionType.JOINT_POSITION,
                 action_prediction_step = 2, action_ori_type = "euler", 
                 observation_type = ObservationType.JOINT_POSITION_ONLY,
                 rotation_transform = None, contain_ft=False,
                 camera_keys=None, camera_keys_transformation=None,
                 state_keys=None, data_type="real_robot",
                 real_robot_tool_sacle=1.0,):
        self.task_dir = task_dir
        self.json_file = json_file
        self.action_type = action_type
        self._obs_type = observation_type
        self._contain_ft = contain_ft
        self._action_prediction_step = action_prediction_step
        self._action_ori_type = action_ori_type
        # None or dict[str, np.ndarray]
        self._rotation_transform = rotation_transform
        self._camera_keys = camera_keys; self._state_keys = state_keys
        self._camera_keys_transformation = camera_keys_transformation
        self._data_type = data_type
        self._tool_scale = real_robot_tool_sacle

    def return_episode_data(self, episode_idx, skip_steps_nums=1):
        # Load episode data on-demand
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            log.warn(f"Episode {episode_idx} data.json not found.")
            return None

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)
            
        # check if async save ft， @TODO: assumING not using async
        search_pattern = os.path.join(episode_dir, "*ft*.json")
        async_ft_files = glob.glob(search_pattern)
        async_save_ft = True if len(async_ft_files) else False
        if async_save_ft: log.info(f'Async save ft files: {async_ft_files}')

        episode_data = []

        # Loop over the data entries and process each one
        counter = 0; data_type_validated = False
        skip_steps_nums = int(skip_steps_nums)
        if skip_steps_nums > self._action_prediction_step:
            self._action_prediction_step = skip_steps_nums
        len_json_file = len(json_file['data'])
        json_data = json_file['data']
        init_ee_poses = {}
        # @TODO: maybe pose-process for time synchronization
        for i, item_data in enumerate(json_file['data']):
            # skip
            if counter % skip_steps_nums != 0:
                counter += 1
                continue
            else: counter += 1
            
            # Process images and other data
            colors, colors_time_stamp = self._process_images(item_data, 'colors', episode_dir)
            if colors is None or len(colors) == 0:
                log.warn(f'Do not get the {i}th color image from {self.task_dir} {episode_dir}, color is None {colors}')
                continue
            depths, depths_time_stamp = self._process_images(item_data, 'depths', episode_dir)
            if depths is None:
                continue
            audios = self._process_audio(item_data, 'audios', episode_dir)
            
            # Append the observation state data in the item_data list
            joint_states = item_data.get("joint_states", {})
            if not data_type_validated:
                if joint_states is None or len(joint_states) == 0:
                    if "robot" in self._data_type:
                        log.info(f'data type: {self._data_type}')
                        raise ValueError(f'{self.task_dir} contains robot data but could not get joint states from dataset')
                else:
                    if "human" in self._data_type:
                        raise ValueError(f'{self.task_dir} contains human data but get joint states from dataset which is not matching!')
                data_type_validated = True
            joint_check = [ObservationType.JOINT_POSITION_ONLY, ObservationType.JOINT_POSITION_END_EFFECTOR]
            if self._obs_type in joint_check:
                if joint_states is None or len(joint_states) == 0:
                    raise ValueError(f'Do not get the {i}th joint state from {self.task_dir} {episode_dir} for {self._obs_type}')
            ee_states = item_data.get('ee_states', {})
            # 拿到当前episode的初始pose
            if len(init_ee_poses) == 0:
                for key, cur_ee_state in ee_states.items():
                    if self._state_keys:
                        if key not in self._state_keys:
                            continue
                        
                    if self._rotation_transform:
                        pose = cur_ee_state["pose"]
                        init_ee_poses[key] = self.apply_rotation_offset(pose, key)      
                        log.info(f'Successfully updated the init ee pose for relative pose calculation {list(init_ee_poses.keys())}')
                    else: init_ee_poses[key] = None
            # @TODO: used for latter head tracker
            contain_head = "head" in ee_states
            ee_check = [ObservationType.JOINT_POSITION_END_EFFECTOR, ObservationType.END_EFFECTOR_POSE,
                        ObservationType.DELTA_END_EFFECTOR_POSE] 
            if self._obs_type in ee_check:
                if ee_states is None or len(ee_states) == 0:
                    raise ValueError(f'Do not get the {i}th ee state pose from {self.task_dir} {episode_dir} for {self._obs_type}')
            # update head to cur ee state, 确保head是在最后一个key
            if contain_head:
                assert self._obs_type in [ObservationType.MASK, ObservationType.END_EFFECTOR_POSE], f"{self._obs_type} not support for contain head"
                assert self.action_type in [ActionType.END_EFFECTOR_POSE, ActionType.END_EFFECTOR_POSE_DELTA], f"{self.action_type} not support for contain head"
                log.debug(f'ee states contain head pose: {list(ee_states.keys())}')
            ee_check.remove(ObservationType.JOINT_POSITION_END_EFFECTOR)
            
            found_obs_keys = []; cur_obs = {}
            for key, cur_ee_state in ee_states.items():
                if self._state_keys:
                    if key not in self._state_keys:
                        continue
                found_obs_keys.append(key)
                
                if self._obs_type in joint_check:
                    cur_obs[key] = np.array(joint_states[key]["position"])
                    if self._obs_type == ObservationType.JOINT_POSITION_END_EFFECTOR:
                        ee_pose = self.apply_rotation_offset(ee_states[key]["pose"], key,
                                                            init_data=init_ee_poses[key])
                        cur_obs[key] = np.hstack((cur_obs[key], ee_pose))
                elif self._obs_type in ee_check:
                    # get cur rotated relative pose
                    ee_pose = self.apply_rotation_offset(ee_states[key]["pose"], key,
                                                        init_data=init_ee_poses[key])
                    if self._obs_type == ObservationType.END_EFFECTOR_POSE:
                        cur_obs[key] = np.array(ee_pose)
                    else:
                        last_id = i - 1
                        last_ee_states = ee_states if last_id < 0 else json_data[last_id].get("ee_states", {}) 
                        last_pose = last_ee_states[key]["pose"]
                        # if has rotation offset, calculate the rotated relative pose 
                        last_pose = self.apply_rotation_offset(last_pose, key,
                                                init_data=init_ee_poses[key])
                        # delta: the diff between two relative pose if has rot offset
                        cur_obs[key] = self.get_pose_diff(ee_pose, last_pose)
                elif self._obs_type == ObservationType.FT_ONLY or self._contain_ft:
                    if async_save_ft:
                        assert len(async_ft_files) == len(list(ee_states.keys())), f'len async save ft files {len(async_ft_files)} != len ee {len(list(ee_states.keys()))}'
                        # @TODO: zyx, process the asyn ft data
                        
                    else:
                        if 'ft' not in cur_ee_state:
                            raise ValueError(f'ee state {key} not contain ft keys')
                        cur_obs[key] = np.array(cur_ee_state["ft"])
                elif self._obs_type == ObservationType.MASK:
                    cur_obs[key] = np.zeros(7)
                    
                if self._contain_ft :
                    if self._obs_type != ObservationType.FT_ONLY:
                        cur_obs[key] = np.hstack((cur_obs[key], cur_ee_state["ft"]))
                    else:
                        log.warn(f'Your obs type is already ft only so no need to contain ft anymore!!!!')
            if self._state_keys is None: self._state_keys = found_obs_keys
            assert len(found_obs_keys) == len(self._state_keys), f"expected {self._state_keys}, but only found {found_obs_keys}"
            
            # Append the action data in the item_data list
            cur_actions = {}
            action_state_id = i+self._action_prediction_step
            if action_state_id >= len_json_file: continue
            if self.action_type == ActionType.JOINT_POSITION:
                joint_states = item_data.get("joint_states", {})
                cur_actions = self._get_absolute_action(joint_states, 
                        action_state=json_data[action_state_id]["joint_states"],
                                                    attribute_name="position")
            elif self.action_type == ActionType.END_EFFECTOR_POSE:
                cur_actions = self._get_absolute_action(item_data.get("ee_states", {}),
                                    action_state=json_data[action_state_id]["ee_states"],
                                    attribute_name="pose", init_data=init_ee_poses)
                # different action rotation representation than quaternion
                for key, action in cur_actions.items():
                    if self._action_ori_type == 'euler':
                        modified_action = np.zeros(6)
                        modified_action[:3] = action[:3]
                        modified_action[3:] = R.from_quat(action[3:]).as_euler("xyz", False)
                    elif self._action_ori_type == "6d_rotation":
                        modified_action = np.zeros(9)
                        modified_action[:3] = action[:3]
                        modified_action[3:] = self.convert_qaut_to_6d_rot(action[3:])
                    elif self._action_ori_type != "quaternion":
                        raise ValueError(f'The action orientation type {self._action_ori_type} is not supported for reading episode data')
                    else: continue
                    cur_actions[key] = modified_action
            elif self.action_type == ActionType.JOINT_POSITION_DELTA:
                joint_states = item_data.get("joint_states", {})
                next_state_data = json_data[action_state_id].get("joint_states", {})
                cur_actions = self._get_delta_action(joint_states, next_state_data, "position")
            elif self.action_type == ActionType.END_EFFECTOR_POSE_DELTA:
                ee_states = item_data.get("ee_states", {})
                next_state_data = json_data[action_state_id].get("ee_states", {})
                for key, pose in ee_states.items():
                    if self._state_keys:
                        if key not in self._state_keys:
                            continue
                        
                    cur_actions[key] = np.zeros(7)
                    next_pose = np.array(next_state_data[key]["pose"])
                    next_pose = self.apply_rotation_offset(next_pose, key, init_ee_poses[key])
                    cur_pose = self.apply_rotation_offset(np.array(pose["pose"]), key, init_ee_poses[key])
                    cur_actions[key] = self.get_pose_diff(next_pose, cur_pose)
                    # different action rotation representation than quaternion
                    if self._action_ori_type == 'euler':
                        modified_action = np.zeros(6)
                        modified_action[:3] = cur_actions[key][:3]
                        modified_action[3:] = R.from_quat(cur_actions[key][3:]).as_euler("xyz", False)
                    elif self._action_ori_type == "6d_rotation":
                        modified_action = np.zeros(9)
                        modified_action[:3] = cur_actions[key][:3]
                        modified_action[3:] = self.convert_qaut_to_6d_rot(cur_actions[key][3:])
                    elif self._action_ori_type != "quaternion":
                        raise ValueError(f'The action orientation type {self._action_ori_type} is not supported for reading episode data')
                    else: continue
                    cur_actions[key] = modified_action
               
            # tool state
            tool_states = item_data.get("tools", {})
            if tool_states is None or len(tool_states) == 0:
                log.warn(f'Dataset do not contain tool infos, please double confirm whether you are training a policy without tool usage!!!')
            else:
                action_tool_states = json_data[action_state_id].get("tools", {})
                for key in cur_actions.keys():
                    if "head" in key: continue
                    tool_scale = self._tool_scale if "robot" in self._data_type else 1.0
                    action_tool_state = action_tool_states[key]["position"] / tool_scale
                    tool_state = tool_states[key]["position"] / tool_scale
                    cur_actions[key] = np.hstack((cur_actions[key], action_tool_state))
                    if self._obs_type == ObservationType.MASK:
                        cur_obs[key] = np.hstack((cur_obs[key], [0]))
                    else:
                        cur_obs[key] = np.hstack((cur_obs[key], tool_state))
                        
            # @TODO: @zyx add the keyframe to the action
            assert len(cur_actions.keys()) == len(self._state_keys), f"expected {self._state_keys}, but only found {list(cur_actions.keys())}"
            # log.info(f'color keys: {list(colors.keys())}')
            # if counter % skip_steps_nums == 0:
            episode_data.append(
                {
                    'idx': item_data.get('idx', 0),
                    'colors': colors,
                    'colors_time_stamp': colors_time_stamp,
                    'depths': depths,
                    'depths_time_stamp': depths_time_stamp,
                    'joint_states': item_data.get('joint_states', {}),
                    'ee_states': item_data.get('ee_states', {}),
                    'tools': item_data.get('tools', {}),
                    'imus': item_data.get('imus', {}),
                    'tactiles': item_data.get('tactiles', {}),
                    'audios': audios,
                    'actions': cur_actions,
                    'observations': cur_obs
                }
            )
        
        return episode_data
    
    def get_episode_text_info(self, episode_id):
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_id:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Episode {episode_id} data.json not found.")

        with open(json_path, 'r', encoding='utf-8') as jsonf:
            json_file = json.load(jsonf)

        text_info = json_file["text"]
        if not "steps" in text_info:
            return None
        
        steps = ""
        if isinstance(text_info["steps"], dict):
            for step_number, cur_step in text_info["steps"].items():
                steps += cur_step
                steps += " "
        else: steps = text_info["steps"]
        
        text_info = 'description: ' + text_info["desc"] + ' ' \
                + 'steps: ' + steps + ' ' + 'goal: ' + text_info["goal"]        
        return text_info
    
    def _get_absolute_action(self, states, action_state, attribute_name = None, init_data = None):
        cur_action = {}
        for key in states.keys():
            if self._state_keys:
                if key not in self._state_keys:
                    continue
                
            if attribute_name is not None:
                if attribute_name == "pose":
                    cur_init_pose = init_data[key] if init_data else None
                    action_state[key][attribute_name] = self.apply_rotation_offset(
                                action_state[key][attribute_name], key, cur_init_pose)
                cur_action[key] = action_state[key][attribute_name]
            else:
                cur_action[key] = action_state[key]
        return cur_action
    
    def _get_delta_action(self, states, next_state_data, attribute_name = None):
        cur_action = {}
        next_state_value = {}
        for key, state in next_state_data.items():
            state_value = state if attribute_name is None else state[attribute_name]
            next_state_value[key] = state_value
        
        for key, state in states.items():
            if self._state_keys:
                if key not in self._state_keys:
                    continue
            state_value = state if attribute_name is None else state[attribute_name]
            cur_action[key] = np.array(next_state_value[key]) - np.array(state_value)
        return cur_action
    
    def get_pose_diff(self, pose1, pose2, posi_translation=True):
        """ pose1 - pose2"""
        pose_diff = np.zeros(7)
        
        rot1 = R.from_quat(pose1[3:])
        rot2 = R.from_quat(pose2[3:])
        rot2_trans = rot2.inv()
        rot = rot2_trans * rot1
        posi_diff = np.array(pose1[:3]) - np.array(pose2[:3])
        if posi_translation:
            pose_diff[:3] = rot2_trans.apply(posi_diff)
        else: pose_diff[:3] = posi_diff
        pose_diff[3:] = rot.as_quat()
        return pose_diff
    
    def convert_quat_to_euler_pose(self, all_ee_states):
        all_ee_states_euler = {}
        # @TODO: attribute name "pose"
        for key, state in all_ee_states.items():
            all_ee_states_euler[key] = np.zeros(6)
            all_ee_states_euler[key][:3] = state["pose"][:3]
            all_ee_states_euler[key][3:] = R.from_quat(state["pose"][3:]).as_euler('xyz', degrees=False)
        return all_ee_states_euler
    
    def convert_qaut_to_6d_rot(self, quat):
        if not isinstance(quat, np.ndarray):
            quat = np.array(quat)
        rot_mat = R.from_quat(quat[..., :4]).as_matrix()
        batch_dim = quat.shape[:-1]
        rot_6d = np.zeros((batch_dim + (6,)))
        rot_6d[..., :6] = rot_mat[..., :2, :].reshape(batch_dim + (6,))
        return rot_6d
    
    def transform_quat(self, quat1, quat2):
        rot_ab = R.from_quat(quat1)
        rot_bc = R.from_quat(quat2)
        rot_ac = rot_ab * rot_bc  # R_ac = R_ab * R_bc
        return rot_ac.as_quat()  # [qx, qy, qz, qw]
    
    def apply_rotation_offset(self, pose, key, init_data = None):
        """
            @brief: Get rotation offseted pose mainly for umi data
                    for specified key and calculate the delta pose
                    if init data is provided as the last para
        """
        if self._rotation_transform is not None:
            # new_pose = copy.deepcopy(pose)
            new_pose = pose.copy()
            if key not in self._rotation_transform:
                # head key rot trans could be unit qunternion(not rotation offset for key)                
                if "head" in key:
                    self._rotation_transform[key] = np.array([0,0,0,1])
                    log.warn("Set the head rotation transform as unit quanternion")
                else:
                    raise ValueError(f'Got the rotation transform but {key} not found in {self._rotation_transform}')
            new_pose[3:] = self.transform_quat(pose[3:], self._rotation_transform[key])
            if init_data is not None:
                # calculate relative term
                new_pose = self.get_pose_diff(new_pose, init_data)
            return new_pose
        else: return pose
        
    def _process_images(self, item_data, data_type, dir_path):
        images = item_data.get(data_type, {})
        if images is None:
            return {}, {}
        
        found_keys = []
        key_transform = self._camera_keys_transformation if "color" in data_type else None
        new_images = {}; time_stamp = {}
        for key, data in images.items():
            if self._camera_keys:
                if key not in self._camera_keys:
                    continue
                
            file_name = data["path"]
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    image = cv2.imread(file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if key_transform is not None and key in key_transform:
                        key = key_transform[key]
                    new_images[key] = image
                    time_stamp[key] = data["time_stamp"]
                    found_keys.append(key)
                else:
                    return None, None
        if self._camera_keys is None:
            self._camera_keys = found_keys
        assert len(found_keys) == len(self._camera_keys), f"expected {self._camera_keys}, but only found {found_keys}"
        return new_images, time_stamp

    def _process_audio(self, item_data, data_type, episode_dir):
        audio_item = item_data.get(data_type, {})
        if audio_item is None:
            return {}
        
        audio_data = {}
        dir_path = os.path.join(episode_dir, data_type)

        for key, file_name in audio_item.items():
            if file_name:
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    pass  # Handle audio data if needed
        return audio_data
    
class LerobotEpisodeReader(RerunEpisodeReader):
    _IMAGE_PREFIX = "observation.images."

    def __init__(self, task_dir = ".", json_file="data.json", 
                 action_type: ActionType = ActionType.JOINT_POSITION,
                 action_prediction_step = 2, action_ori_type = "euler", 
                 observation_type = ObservationType.JOINT_POSITION_ONLY,
                 rotation_transform = None, contain_ft=False,
                 camera_keys=None, camera_keys_transformation=None,
                 state_keys=None, data_type="real_robot",
                 real_robot_tool_sacle=1.0,):
        super().__init__(task_dir, json_file, action_type, action_prediction_step, action_ori_type,
                         observation_type, rotation_transform, contain_ft, camera_keys, camera_keys_transformation,
                         state_keys, data_type, real_robot_tool_sacle)
        if self._contain_ft or self._obs_type == ObservationType.FT_ONLY:
            raise ValueError("LerobotEpisodeReader does not support FT observations in lerobot_v3 data.")

        self._lerobot_dataset = LeRobotV3Dataset(self.task_dir)
        self._episode_starts = [int(x) for x in self._lerobot_dataset.episode_data_index["from"]]
        self._episode_ends = [int(x) for x in self._lerobot_dataset.episode_data_index["to"]]
        self._episode_count = len(self._episode_starts)
        self._camera_feature_map = self._build_camera_feature_map()
        self._episode_tasks = self._build_episode_tasks()
        self._direct_action_available = any(
            key in self._lerobot_dataset.features
            for key in ("action", "action.ee_pose", "action.joint_position", "action.gripper_width")
        )
        if self._direct_action_available:
            log.info(
                "LerobotEpisodeReader reads actions from lerobot_v3 action fields directly; "
                "action_prediction_step is only used as a fallback when action fields are missing."
            )

        if self._camera_keys is None:
            self._camera_keys = list(self._camera_feature_map.keys())
        else:
            self._camera_keys = [key for key in self._camera_keys if key in self._camera_feature_map]

        if self._state_keys is None:
            inferred_state_key = "single"
            if self._rotation_transform is not None and len(self._rotation_transform) == 1:
                inferred_state_key = next(iter(self._rotation_transform))
            self._state_keys = [inferred_state_key]
        elif len(self._state_keys) != 1:
            raise ValueError(
                "LerobotEpisodeReader currently supports exactly one state key because the "
                "lerobot_v3 converter flattens one primary stream."
            )

        self._state_key = self._state_keys[0]
        if self._rotation_transform is None:
            self._rotation_key = self._state_key
        elif self._state_key in self._rotation_transform:
            self._rotation_key = self._state_key
        elif len(self._rotation_transform) == 1:
            self._rotation_key = next(iter(self._rotation_transform))
        else:
            raise ValueError(
                f"Could not match state key {self._state_key!r} to rotation keys "
                f"{list(self._rotation_transform.keys())}."
            )

    def close(self):
        if hasattr(self, "_lerobot_dataset") and self._lerobot_dataset is not None:
            self._lerobot_dataset.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_camera_feature_map(self):
        camera_feature_map = {}
        for feature_name, spec in self._lerobot_dataset.features.items():
            if not feature_name.startswith(self._IMAGE_PREFIX):
                continue
            if spec.get("dtype") != "video":
                continue
            camera_key = feature_name[len(self._IMAGE_PREFIX):]
            camera_feature_map[camera_key] = feature_name
        return camera_feature_map

    def _build_episode_tasks(self):
        episode_tasks = getattr(self._lerobot_dataset, "episode_tasks", None)
        if episode_tasks is None:
            return [""] * self._episode_count
        return list(episode_tasks)

    @staticmethod
    def _feature_vector(sample, feature_name, dtype=np.float32, fallback_key=None, fallback_slice=None):
        if feature_name in sample:
            array = np.asarray(sample[feature_name], dtype=dtype)
        elif fallback_key is not None and fallback_key in sample:
            array = np.asarray(sample[fallback_key], dtype=dtype)
            if fallback_slice is not None:
                array = array[fallback_slice]
        else:
            return None

        if array.ndim == 0:
            array = array.reshape(1)
        return array.reshape(-1)

    @staticmethod
    def _feature_scalar(sample, feature_name, default=np.nan):
        if feature_name not in sample:
            return float(default)
        array = np.asarray(sample[feature_name]).reshape(-1)
        if array.size == 0:
            return float(default)
        return float(array[0])

    def _require_vector(self, sample, feature_name, context, dtype=np.float32, fallback_key=None, fallback_slice=None):
        array = self._feature_vector(
            sample,
            feature_name,
            dtype=dtype,
            fallback_key=fallback_key,
            fallback_slice=fallback_slice,
        )
        if array is None:
            raise ValueError(f"Missing {context} in lerobot_v3 dataset under {self.task_dir}.")
        return array

    def _extract_joint_position(self, sample):
        return self._require_vector(
            sample,
            "observation.state.joint_position",
            "joint position",
            fallback_key="observation.state",
            fallback_slice=slice(7, 14),
        )

    def _extract_ee_pose(self, sample):
        return self._require_vector(
            sample,
            "observation.state.ee_pose",
            "end-effector pose",
            fallback_key="observation.state",
            fallback_slice=slice(0, 7),
        )

    def _extract_gripper_width(self, sample):
        return self._feature_vector(
            sample,
            "observation.state.gripper_width",
            fallback_key="observation.state",
            fallback_slice=slice(14, 15),
        )

    def _extract_action_joint_position(self, sample):
        return self._feature_vector(
            sample,
            "action.joint_position",
            fallback_key="action",
            fallback_slice=slice(7, 14),
        )

    def _extract_action_ee_pose(self, sample):
        return self._feature_vector(
            sample,
            "action.ee_pose",
            fallback_key="action",
            fallback_slice=slice(0, 7),
        )

    def _extract_action_gripper_width(self, sample):
        return self._feature_vector(
            sample,
            "action.gripper_width",
            fallback_key="action",
            fallback_slice=slice(14, 15),
        )

    def _extract_colors(self, sample):
        colors = {}
        colors_time_stamp = {}
        for camera_key in self._camera_keys:
            feature_name = self._camera_feature_map.get(camera_key)
            if feature_name is None or feature_name not in sample:
                continue

            output_key = camera_key
            if self._camera_keys_transformation is not None and camera_key in self._camera_keys_transformation:
                output_key = self._camera_keys_transformation[camera_key]

            image = np.asarray(sample[feature_name])
            if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
                image = np.moveaxis(image, 0, -1)
            if np.issubdtype(image.dtype, np.floating) and image.size > 0 and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
            colors[output_key] = image
            colors_time_stamp[output_key] = self._feature_scalar(
                sample,
                f"{feature_name}.timestamp",
            )
        return colors, colors_time_stamp

    def _get_episode_bounds(self, episode_idx):
        if episode_idx < 0 or episode_idx >= self._episode_count:
            raise IndexError(f"Episode index {episode_idx} out of range [0, {self._episode_count - 1}]")
        return self._episode_starts[episode_idx], self._episode_ends[episode_idx]

    def _make_joint_state(self, position, timestamp):
        return {
            self._state_key: {
                "position": np.asarray(position, dtype=np.float32),
                "time_stamp": float(timestamp),
            }
        }

    def _make_ee_state(self, pose, timestamp):
        return {
            self._state_key: {
                "pose": np.asarray(pose, dtype=np.float32),
                "time_stamp": float(timestamp),
            }
        }

    def _make_tool_state(self, gripper_width, timestamp):
        if gripper_width is None:
            return {}
        tool_value = np.asarray(gripper_width, dtype=np.float32).reshape(-1)
        if tool_value.size == 0:
            return {}
        return {
            self._state_key: {
                "position": float(tool_value[0]),
                "time_stamp": float(timestamp),
            }
        }

    def _build_observation(self, current_joint, current_ee, previous_ee, init_ee_pose, current_tool):
        joint_check = [ObservationType.JOINT_POSITION_ONLY, ObservationType.JOINT_POSITION_END_EFFECTOR]
        ee_check = [
            ObservationType.JOINT_POSITION_END_EFFECTOR,
            ObservationType.END_EFFECTOR_POSE,
            ObservationType.DELTA_END_EFFECTOR_POSE,
        ]

        if self._obs_type in joint_check:
            observation = current_joint.astype(np.float32, copy=True)
            if self._obs_type == ObservationType.JOINT_POSITION_END_EFFECTOR:
                ee_pose = self.apply_rotation_offset(
                    current_ee,
                    self._rotation_key,
                    init_data=init_ee_pose,
                )
                observation = np.hstack((observation, ee_pose))
        elif self._obs_type in ee_check:
            ee_pose = self.apply_rotation_offset(
                current_ee,
                self._rotation_key,
                init_data=init_ee_pose,
            )
            if self._obs_type == ObservationType.END_EFFECTOR_POSE:
                observation = np.asarray(ee_pose, dtype=np.float32)
            else:
                last_pose = self.apply_rotation_offset(
                    previous_ee,
                    self._rotation_key,
                    init_data=init_ee_pose,
                )
                observation = self.get_pose_diff(ee_pose, last_pose).astype(np.float32)
        elif self._obs_type == ObservationType.MASK:
            observation = np.zeros(7, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported observation type for lerobot_v3 data: {self._obs_type}")

        if current_tool is not None:
            tool_scale = self._tool_scale if "robot" in self._data_type else 1.0
            tool_value = np.asarray(current_tool, dtype=np.float32).reshape(-1) / tool_scale
            if self._obs_type == ObservationType.MASK:
                observation = np.hstack((observation, [0]))
            else:
                observation = np.hstack((observation, tool_value))

        return {self._state_key: observation.astype(np.float32, copy=False)}

    def _convert_action_orientation(self, action):
        if self._action_ori_type == "euler":
            modified_action = np.zeros(6, dtype=np.float32)
            modified_action[:3] = action[:3]
            modified_action[3:] = R.from_quat(action[3:]).as_euler("xyz", False)
            return modified_action
        if self._action_ori_type == "6d_rotation":
            modified_action = np.zeros(9, dtype=np.float32)
            modified_action[:3] = action[:3]
            modified_action[3:] = self.convert_qaut_to_6d_rot(action[3:])
            return modified_action
        if self._action_ori_type == "quaternion":
            return action.astype(np.float32, copy=False)
        raise ValueError(
            f"The action orientation type {self._action_ori_type} is not supported for lerobot_v3 data."
        )

    def _build_action(self, current_joint, current_ee, current_tool, action_sample, future_sample, init_ee_pose):
        action_joint = self._extract_action_joint_position(action_sample)
        action_ee = self._extract_action_ee_pose(action_sample)
        action_tool = self._extract_action_gripper_width(action_sample)

        if action_joint is None and future_sample is not None:
            action_joint = self._extract_joint_position(future_sample)
        if action_ee is None and future_sample is not None:
            action_ee = self._extract_ee_pose(future_sample)
        if action_tool is None and future_sample is not None:
            action_tool = self._extract_gripper_width(future_sample)

        if self.action_type == ActionType.JOINT_POSITION:
            if action_joint is None:
                raise ValueError(f"Missing joint action in lerobot_v3 dataset under {self.task_dir}.")
            action = action_joint.astype(np.float32, copy=False)
        elif self.action_type == ActionType.END_EFFECTOR_POSE:
            if action_ee is None:
                raise ValueError(f"Missing end-effector action in lerobot_v3 dataset under {self.task_dir}.")
            action = self.apply_rotation_offset(
                action_ee,
                self._rotation_key,
                init_data=init_ee_pose,
            ).astype(np.float32, copy=False)
            action = self._convert_action_orientation(action)
        elif self.action_type == ActionType.JOINT_POSITION_DELTA:
            if action_joint is None:
                raise ValueError(f"Missing joint action in lerobot_v3 dataset under {self.task_dir}.")
            action = (np.asarray(action_joint) - np.asarray(current_joint)).astype(np.float32)
        elif self.action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            if action_ee is None:
                raise ValueError(f"Missing end-effector action in lerobot_v3 dataset under {self.task_dir}.")
            next_pose = self.apply_rotation_offset(
                action_ee,
                self._rotation_key,
                init_data=init_ee_pose,
            )
            cur_pose = self.apply_rotation_offset(
                current_ee,
                self._rotation_key,
                init_data=init_ee_pose,
            )
            action = self.get_pose_diff(next_pose, cur_pose).astype(np.float32)
            action = self._convert_action_orientation(action)
        else:
            raise ValueError(f"Unsupported action type for lerobot_v3 data: {self.action_type}")

        if action_tool is not None:
            tool_scale = self._tool_scale if "robot" in self._data_type else 1.0
            tool_value = np.asarray(action_tool, dtype=np.float32).reshape(-1) / tool_scale
            action = np.hstack((action, tool_value))

        return {self._state_key: action.astype(np.float32, copy=False)}

    def return_episode_data(self, episode_idx, skip_steps_nums=1):
        episode_start, episode_end = self._get_episode_bounds(int(episode_idx))
        skip_steps_nums = max(1, int(skip_steps_nums))
        effective_action_step = max(skip_steps_nums, int(self._action_prediction_step))
        episode_data = []
        previous_ee = None
        init_ee_pose = None

        for global_frame_index in range(episode_start, episode_end, skip_steps_nums):
            sample = self._lerobot_dataset[global_frame_index]
            current_joint = self._extract_joint_position(sample)
            current_ee = self._extract_ee_pose(sample)
            current_tool = self._extract_gripper_width(sample)
            current_timestamp = self._feature_scalar(sample, "timestamp")

            if init_ee_pose is None:
                init_ee_pose = (
                    self.apply_rotation_offset(current_ee, self._rotation_key)
                    if self._rotation_transform is not None
                    else None
                )
            if previous_ee is None:
                previous_ee = current_ee.copy()

            colors, colors_time_stamp = self._extract_colors(sample)
            future_sample = None
            future_index = global_frame_index + effective_action_step
            if not self._direct_action_available:
                if future_index >= episode_end:
                    continue
                future_sample = self._lerobot_dataset[future_index]

            joint_states = self._make_joint_state(current_joint, current_timestamp)
            ee_states = self._make_ee_state(current_ee, current_timestamp)
            tool_states = self._make_tool_state(current_tool, current_timestamp)

            cur_obs = self._build_observation(
                current_joint=current_joint,
                current_ee=current_ee,
                previous_ee=previous_ee,
                init_ee_pose=init_ee_pose,
                current_tool=current_tool,
            )
            cur_actions = self._build_action(
                current_joint=current_joint,
                current_ee=current_ee,
                current_tool=current_tool,
                action_sample=sample,
                future_sample=future_sample,
                init_ee_pose=init_ee_pose,
            )

            local_idx = int(self._feature_scalar(sample, "frame_index", default=global_frame_index - episode_start))
            episode_data.append(
                {
                    "idx": local_idx,
                    "colors": colors,
                    "colors_time_stamp": colors_time_stamp,
                    "depths": {},
                    "depths_time_stamp": {},
                    "joint_states": joint_states,
                    "ee_states": ee_states,
                    "tools": tool_states,
                    "imus": {},
                    "tactiles": {},
                    "audios": {},
                    "actions": cur_actions,
                    "observations": cur_obs,
                }
            )
            previous_ee = current_ee.copy()

        return episode_data

    def get_episode_text_info(self, episode_id):
        episode_id = int(episode_id)
        if episode_id < 0 or episode_id >= self._episode_count:
            raise IndexError(f"Episode index {episode_id} out of range [0, {self._episode_count - 1}]")
        task_text = self._episode_tasks[episode_id] if episode_id < len(self._episode_tasks) else ""
        if not task_text:
            return None
        return f"description: {task_text}"

if __name__ == "__main__":
    # episode_reader = RerunEpisodeReader(task_dir = unzip_file_output_dir)
    # # TEST EXAMPLE 1 : OFFLINE DATA TEST
    # episode_data6 = episode_reader.return_episode_data(6)
    # logger_mp.info("Starting offline visualization...")
    # offline_logger = RerunLogger(prefix="offline/")
    # offline_logger.log_episode_data(episode_data6)
    # logger_mp.info("Offline visualization completed.")
    
    data_folder = "dataset/data/test_now"
    cur_path = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(cur_path, '../..', data_folder)
    episode_reader = RerunEpisodeReader(task_dir=task_dir, action_type=ActionType.JOINT_POSITION_DELTA)
    data = episode_reader.return_episode_data(2, 1)
    print(f'data: {data}')    
