"""
    Contributions to the unitree robotics xr_teleoperation projects
    with modification by ZYX
"""

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
from reader import RerunEpisodeReader, LerobotEpisodeReader, ActionType, ObservationType
from datetime import datetime
import logging_mp
os.environ["RUST_LOG"] = "error"

# Initialize logger for the module
logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

def transform_pose(pose1, pose2):
        pose = np.zeros(7)
        t1 = pose1[:3]; t2 = pose2[:3]
                
        rot1 = R.from_quat(pose1[3:])
        rot2 = R.from_quat(pose2[3:])
        pose[:3] = t1 + rot1.apply(t2)
        pose[3:] = (rot1 * rot2).as_quat()
        return pose


def build_episode_reader(task_dir, data_record_type="hirol", **reader_kwargs):
    data_record_type = str(data_record_type).lower()
    if data_record_type == "hirol":
        reader_cls = RerunEpisodeReader
    elif data_record_type in {"lerobotv3", "lerobot_v3"}:
        reader_cls = LerobotEpisodeReader
    else:
        raise ValueError(
            f"Unsupported data_record_type={data_record_type!r}. Expected 'hirol' or 'lerobotv3'."
        )

    logger_mp.info(f"Using {reader_cls.__name__} for data_record_type={data_record_type}")
    return reader_cls(task_dir=task_dir, **reader_kwargs)
    
class RerunLogger:
    def __init__(self, task_dir, prefix = "", IdxRangeBoundary = 30, memory_limit = None, 
                 example_item_data = None, action_type = ActionType.JOINT_POSITION, 
                 action_ori_type="quaternion", state_keys=None):
        self._task_dir = task_dir
        self.prefix = prefix
        self._action_type = action_type
        self._action_ori_type = action_ori_type
        self.IdxRangeBoundary = IdxRangeBoundary
        self.ee_trajectories = {}  # Store ee trajectories (positions) for each ee_key
        self.ee_orientations = {}  # Store ee orientations (quaternions) for each ee_key
        self.state_keys = state_keys
        
        rr.init(datetime.now().strftime("Runtime_%Y%m%d_%H%M%S"))
        if memory_limit:
            rr.spawn(memory_limit = memory_limit, hide_welcome_screen = True)
        else:
            rr.spawn(hide_welcome_screen = True)

        # Set up blueprint for live visualization
        if self.IdxRangeBoundary and example_item_data:
            logger_mp.info(f'Setting blue print!!!!!')
            self.setup_blueprint(example_item_data)
        self.counter = 0

    def setup_blueprint(self, example_item_data):
        obs_states_views = []
        action_views = []
        action_pose_views = []
        ee_states_views = []
        image_views = []

        # Dynamically create observation states views based on example data
        obs_states = example_item_data.get('observations', {})
        if obs_states is not None:
            for obs_key in obs_states.keys():
                logger_mp.info(f'obs state shape: {obs_states[obs_key].shape}')
                obs_view = rrb.TimeSeriesView(
                    origin = f"{self.prefix}{obs_key}/obs_states/state",
                    plot_legend = rrb.PlotLegend(visible = True),
                )
                obs_states_views.append(obs_view)
                logger_mp.info(f'Created joint states view for: {obs_key}')

        # Dynamically create action views based on example data
        actions = example_item_data.get('actions', {})
        for action_key in actions.keys():
            if actions[action_key] is not None:
                action_view = rrb.TimeSeriesView(
                    origin = f"{self.prefix}{action_key}/actions/action",
                    plot_legend = rrb.PlotLegend(visible = True),
                )
                action_views.append(action_view)
                logger_mp.info(f'Created action view for: {action_key}')
        
        if self._action_type == ActionType.END_EFFECTOR_POSE or self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            for action_key in actions.keys():
                action_pose_view = rrb.Spatial3DView(
                    origin = f"{self.prefix}{action_key}_action_pose/actions/pose",
                )
                action_pose_views.append(action_pose_view)
                logger_mp.info(f'Created action view for: {action_key} pose')

        # Dynamically create ee_states views based on example data
        ee_states = example_item_data.get('ee_states', {})
        for ee_key in ee_states.keys():
            if ee_key not in self.state_keys:
                continue
            
            # Create 3D spatial view for end-effector pose
            ee_spatial_view = rrb.Spatial3DView(
                origin = f"{self.prefix}{ee_key}/ee_states/pose",
            )
            ee_states_views.append(ee_spatial_view)
            logger_mp.info(f'Created ee_states spatial view for: {ee_key}')

        # Dynamically create image views based on example data
        colors = example_item_data.get('colors', {})
        for color_key in colors.keys():
            if colors[color_key] is not None:
                image_view = rrb.Spatial2DView(
                    origin = f"{self.prefix}colors/{color_key}",
                )
                image_views.append(image_view)
                logger_mp.info(f'Created image view for: {color_key}')

        # Organize views: @TODO: Dynamic assignment
        actions_column = rrb.Vertical(contents=action_views)
        ee_states_row = rrb.Horizontal(contents=ee_states_views)
        if self._action_type == ActionType.END_EFFECTOR_POSE or self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            action_pose_row = rrb.Horizontal(contents=action_pose_views)
        images_column = rrb.Vertical(contents=image_views)

        obs_states_column = rrb.Vertical(contents=obs_states_views)
        first_row = rrb.Horizontal(contents=[obs_states_column, actions_column])
        if self._action_type == ActionType.END_EFFECTOR_POSE or self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            first_col = rrb.Vertical(contents=[first_row, action_pose_row, ee_states_row])
        else:
            first_col = rrb.Vertical(contents=[first_row, ee_states_row])
        grid = rrb.Horizontal(contents=[first_col, images_column])
        
        # Add control panels
        blueprint_content = [
            grid,
            rr.blueprint.SelectionPanel(state=rrb.PanelState.Collapsed),
            rr.blueprint.TimePanel(state=rrb.PanelState.Collapsed)
        ]
        rr.send_blueprint(rrb.Vertical(contents=blueprint_content))

    
    def _log_trajectory_with_orientation(self, ee_key, log_name):
        """Log trajectory with orientation visualization using coordinate axes"""
        positions = np.array(self.ee_trajectories[ee_key])
        orientations = np.array(self.ee_orientations[ee_key])
        
        # Log trajectory line in the pose space
        if len(positions) > 1:
            rr.log(f"{self.prefix}{log_name}/trajectory_line", 
                   rr.LineStrips3D([positions], colors=[0, 255, 255], radii=[0.002]))  # Cyan trajectory
        
        # Log coordinate axes at each point to show orientation
        if len(positions) >= 1:
            # Sample points for orientation visualization (every 3rd point to avoid clutter)
            sample_rate = max(1, len(positions) // 15)  # Show at most 15 coordinate frames
            
            for i in range(0, len(positions), sample_rate):
                pos = positions[i]
                quat = orientations[i]
                
                try:
                    # Convert quaternion to rotation matrix and get axis directions
                    rot = R.from_quat(quat)
                    axis_length = 0.03  # 3cm axes for trajectory points
                    
                    # Create local coordinate axes vectors
                    x_axis = rot.apply([axis_length, 0, 0])  # Local X axis
                    y_axis = rot.apply([0, axis_length, 0])  # Local Y axis
                    z_axis = rot.apply([0, 0, axis_length])  # Local Z axis
                    
                    # Log coordinate axes at this trajectory point (without background labels)
                    rr.log(f"{self.prefix}{log_name}/trajectory_orientations/point_{i}_axes", 
                           rr.Arrows3D(
                               origins=[pos, pos, pos],
                               vectors=[x_axis, y_axis, z_axis],
                               colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red X, Green Y, Blue Z
                           ))
                    
                    # Color-coded arrows show orientation: Red=X, Green=Y, Blue=Z
                    
                except Exception as e:
                    logger_mp.warning(f'Failed to create coordinate axes for point {i}: {e}')
        
        # Log current position as a highlighted point in pose space
        if len(positions) >= 1:
            current_pos = positions[-1]
            rr.log(f"{self.prefix}{log_name}/current_position", 
                   rr.Points3D(positions=[current_pos], colors=[255, 0, 255], radii=[0.008]))  # Magenta point
        
        # Also log all trajectory points for better visualization
        rr.log(f"{self.prefix}{log_name}/trajectory_points", 
               rr.Points3D(positions=positions, colors=[100, 255, 100], radii=[0.003]))  # Light green points

    def log_item_data(self, item_data: dict):
        rr.set_time_sequence("idx", self.counter)
        self.counter += 1

        # Log states
        states = item_data.get('observations', {}) or {}
        for key, obs_state in states.items():
            logger_mp.info(f'{key} obs_state: {obs_state.shape}')
            rr.log(f"{self.prefix}{key}/obs_states/state", rr.Scalar(obs_state))

        # Log actions
        actions = item_data.get('actions', {}) or {}
        for action_key, action_val in actions.items():
            if action_val is not None:
                logger_mp.info(f'Logging action {action_key} with shape: {action_val.shape if hasattr(action_val, "shape") else type(action_val)}')
                rr.log(f"{self.prefix}{action_key}/actions/action", rr.Scalar(action_val[-1]))
            else:
                logger_mp.warning(f'Could not find {action_key} for action_key')
                
        # log action pose
        ee_states = item_data.get('ee_states', {}) or {}
        if self._action_type == ActionType.END_EFFECTOR_POSE or self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
            for action_key, action_val in actions.items():
                original_key = action_key
                action_key = action_key + "_action_pose"
                pose_dim = 7 if self._action_ori_type == "quaternion" else 6
                action_val = action_val[:pose_dim]
                if self._action_ori_type == "euler":
                    action_val = R.from_euler("xyz", action_val).as_quat()
                if self._action_type == ActionType.END_EFFECTOR_POSE_DELTA:
                    action_val = transform_pose(ee_states[original_key]["pose"], action_val)
                
                position = np.array(action_val[:3])  # First 3 elements: position
                quaternion = np.array(action_val[3:7])  # Next 4 elements: quaternion (qx, qy, qz, qw)
                
                # Validate position and quaternion
                if not np.all(np.isfinite(position)):
                    logger_mp.warning(f'Invalid position values for {action_key}: {position}')
                    continue
                if not np.all(np.isfinite(quaternion)):
                    logger_mp.warning(f'Invalid quaternion values for {action_key}: {quaternion}')
                    continue
                
                # Normalize quaternion to ensure it's valid
                quat_norm = np.linalg.norm(quaternion)
                if quat_norm > 0:
                    quaternion = quaternion / quat_norm
                else:
                    logger_mp.warning(f'Zero quaternion for {action_key}, using identity quaternion')
                    quaternion = np.array([0, 0, 0, 1])  # Identity quaternion
                
                # Initialize trajectory storage if needed
                if action_key not in self.ee_trajectories:
                    self.ee_trajectories[action_key] = []
                    self.ee_orientations[action_key] = []
                
                # Add current position and orientation to trajectory
                self.ee_trajectories[action_key].append(position.copy())
                self.ee_orientations[action_key].append(quaternion.copy())
                
                # Log coordinate axes at origin in the pose space
                axis_length = 0.1  # 10cm axes
                rr.log(f"{self.prefix}{action_key}/actions/pose/coordinate_axes", 
                        rr.Arrows3D(
                            origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
                            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red X, Green Y, Blue Z
                        ))
                
                # Color-coded arrows are self-explanatory: Red=X, Green=Y, Blue=Z
                
                # Log current 3D pose with position and rotation
                rr.log(f"{self.prefix}{action_key}/actions/pose", 
                        rr.Transform3D(
                            translation=position,
                            rotation=rr.Quaternion(xyzw=quaternion),
                        ))
                
                # Log end-effector coordinate frame at current position
                rr.log(f"{self.prefix}{action_key}/actions/pose/ee_frame", 
                        rr.Arrows3D(
                            origins=[position, position, position],
                            vectors=[
                                [axis_length, 0, 0],  # X-axis
                                [0, axis_length, 0],  # Y-axis  
                                [0, 0, axis_length]   # Z-axis
                            ],
                            colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]] # Light RGB
                        ))
                
                # Light color-coded arrows for current EE frame: Light Red=X, Light Green=Y, Light Blue=Z
                
                # Log trajectory with orientation visualization
                if len(self.ee_trajectories[action_key]) >= 1:
                    self._log_trajectory_with_orientation(action_key, f'{action_key}/actions/pose')
                    logger_mp.info(f'Logged {len(self.ee_trajectories[action_key])} trajectory points for {action_key} {self._action_type}')
            
        # Log ee_states (end-effector states) - 7D array: position (3D) + quaternion (4D)
        for ee_key, ee_state in ee_states.items():
            if ee_key not in self.state_keys:
                continue
            
            if ee_state is not None:
                ee_state = ee_state["pose"]
                logger_mp.info(f'Logging ee_state {ee_key} with shape: {ee_state.shape if hasattr(ee_state, "shape") else type(ee_state)}')
                # logger_mp.info(f'EE state values - Position: {ee_state[:3]}, Quaternion: {ee_state[3:7]}')
                # ee_state should be a 7D numpy array: [x, y, z, qx, qy, qz, qw]
                if hasattr(ee_state, '__len__') and len(ee_state) >= 7:
                    position = np.array(ee_state[:3])  # First 3 elements: position
                    quaternion = np.array(ee_state[3:7])  # Next 4 elements: quaternion (qx, qy, qz, qw)
                    
                    # Validate position and quaternion
                    if not np.all(np.isfinite(position)):
                        logger_mp.warning(f'Invalid position values for {ee_key}: {position}')
                        continue
                    if not np.all(np.isfinite(quaternion)):
                        logger_mp.warning(f'Invalid quaternion values for {ee_key}: {quaternion}')
                        continue
                    
                    # Normalize quaternion to ensure it's valid
                    quat_norm = np.linalg.norm(quaternion)
                    if quat_norm > 0:
                        quaternion = quaternion / quat_norm
                    else:
                        logger_mp.warning(f'Zero quaternion for {ee_key}, using identity quaternion')
                        quaternion = np.array([0, 0, 0, 1])  # Identity quaternion
                    
                    # Initialize trajectory storage if needed
                    if ee_key not in self.ee_trajectories:
                        self.ee_trajectories[ee_key] = []
                        self.ee_orientations[ee_key] = []
                    
                    # Add current position and orientation to trajectory
                    self.ee_trajectories[ee_key].append(position.copy())
                    self.ee_orientations[ee_key].append(quaternion.copy())
                    
                    # Log coordinate axes at origin in the pose space
                    axis_length = 0.1  # 10cm axes
                    rr.log(f"{self.prefix}{ee_key}/ee_states/pose/coordinate_axes", 
                           rr.Arrows3D(
                               origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                               vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
                               colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # Red X, Green Y, Blue Z
                           ))
                    
                    # Color-coded arrows are self-explanatory: Red=X, Green=Y, Blue=Z
                    
                    # Log current 3D pose with position and rotation
                    rr.log(f"{self.prefix}{ee_key}/ee_states/pose", 
                           rr.Transform3D(
                               translation=position,
                               rotation=rr.Quaternion(xyzw=quaternion),
                           ))
                    
                    # Log end-effector coordinate frame at current position
                    rr.log(f"{self.prefix}{ee_key}/ee_states/pose/ee_frame", 
                           rr.Arrows3D(
                               origins=[position, position, position],
                               vectors=[
                                   [axis_length, 0, 0],  # X-axis
                                   [0, axis_length, 0],  # Y-axis  
                                   [0, 0, axis_length]   # Z-axis
                               ],
                               colors=[[255, 100, 100], [100, 255, 100], [100, 100, 255]] # Light RGB
                           ))
                    
                    # Light color-coded arrows for current EE frame: Light Red=X, Light Green=Y, Light Blue=Z
                    
                    # Log trajectory with orientation visualization
                    if len(self.ee_trajectories[ee_key]) >= 1:
                        self._log_trajectory_with_orientation(ee_key, f'{ee_key}/ee_states/pose')
                        logger_mp.info(f'Logged {len(self.ee_trajectories[ee_key])} trajectory points for {ee_key} in ee states')
                           
                else:
                    logger_mp.warning(f'Invalid ee_state format for {ee_key}: expected 7D array, got {ee_state}')
            else:
                logger_mp.warning(f'Could not find {ee_key} for ee_key')

        # Log colors (images) - dynamically support all available images
        colors = item_data.get('colors', {}) or {}
        logger_mp.info(f'Colors data: {list(colors.keys())}')
        for color_key, color_val in colors.items():
            if color_val is not None:
                logger_mp.info(f'Logging image {color_key} with shape: {color_val.shape if hasattr(color_val, "shape") else type(color_val)}')
                rr.log(f"{self.prefix}colors/{color_key}", rr.Image(color_val))
            else:
                logger_mp.warning(f'Could not find {color_key} for color_key')

        # # Log depths (images)
        # depths = item_data.get('depths', {}) or {}
        # for depth_key, depth_val in depths.items():
        #     if depth_val is not None:
        #         # rr.log(f"{self.prefix}depths/{depth_key}", rr.Image(depth_val))
        #         pass # Handle depth if needed

        # # Log tactile if needed
        # tactiles = item_data.get('tactiles', {}) or {}
        # for hand, tactile_vals in tactiles.items():
        #     if tactile_vals is not None:
        #         pass # Handle tactile if needed

    def log_episode_data(self, episode_data: list):
        for item_data in episode_data:
            self.log_item_data(item_data)

if __name__ == "__main__":


    # # TEST DATA OF data_dir
    # dataset/data/1212_duo_unitree_bread_n_picking——214ep dataset/data/1221_duo_unitree_bread_picking_human_114ep
    data_dir = "/home/tele/Code/das-traj-generation/data/hirol_0310_100ep"
    episode_data_number = 69
    fps = 100
    skip_step_nums = 1
    action_ori_type = "quaternion"
    episode_dir = f"episode_{str(episode_data_number).zfill(4)}"
    # umi_rotation_transform = {"single": [0.7071068, 0, 0.7071068, 0]}
    umi_rotation_transform = {"left": [0, 0, 0, 1], "right": [0, 0, 0, 1], "head": [0, 0, 0, 1]}
    umi_rotation_transform = None
    camera_keys = ['head_color', 'left_hand_fisheye_color', 'right_hand_fisheye_color']
    cam_key_transform = {"left_hand_fisheye_color": "left_fisheye_color",
                         "right_hand_fisheye_color": "right_fisheye_color"}
    # camera_keys = ['left', 'mid', 'right']
    # cam_key_transform = {"left": "left_color",
    #                      "right": "right_color",
    #                      "mid": " mid_color"}
    contain_ft = False; data_type = "human_hand"
    if os.path.exists(os.path.join(data_dir, episode_dir)):
        logger_mp.info(f'Found the {episode_dir} in {data_dir}')
        # camera_keys=["right_hand_color", "right_hand_fisheye_color"],
                                            #  state_keys=["right"]
        episode_reader = build_episode_reader(
            task_dir=data_dir,
            data_record_type="hirol",
            action_type=ActionType.END_EFFECTOR_POSE,
            action_prediction_step=1,
            action_ori_type=action_ori_type,
            observation_type=ObservationType.END_EFFECTOR_POSE,
            rotation_transform=umi_rotation_transform,
            contain_ft=contain_ft,
            camera_keys=None,
            camera_keys_transformation=None,
            #  camera_keys=["right_hand_color", "right_hand_fisheye_color"],
            #  state_keys=["right"]
            data_type=data_type,
            real_robot_tool_sacle=0.96,
        )
        episode_data = episode_reader.return_episode_data(episode_data_number, skip_step_nums)
        logger_mp.info(f'Successfully load the episode data')
        logger_mp.info(f'Episode data length: {len(episode_data) if episode_data else 0}')
        # Use first episode data as example for blueprint setup
        example_data = episode_data[0] if episode_data else None
        state_keys = episode_reader._state_keys
        online_logger = RerunLogger(task_dir=data_dir, prefix="offline/", IdxRangeBoundary = 60, memory_limit="20GB",
                example_item_data=example_data, action_type=ActionType.END_EFFECTOR_POSE, state_keys=state_keys)
        logger_mp.info(f'Starting to log {len(episode_data)} items...')
        for item_data in episode_data:
            # logger_mp.info(f'item data: {item_data}')
            online_logger.log_item_data(item_data)
            time.sleep(1/fps) # 30hz
        logger_mp.info("Offline visualization completed.")
    else:
        logger_mp.warning(f'Could not find {data_dir}/{episode_dir}')
        
