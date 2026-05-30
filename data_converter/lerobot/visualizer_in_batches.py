"""
    Contributions to the unitree robotics xr_teleoperation projects
    with modification by ZYX
"""

import os
os.environ["RUST_LOG"] = "off"
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import rerun as rr
import rerun.blueprint as rrb
from reader import ActionType, ObservationType, LerobotEpisodeReader
from rerun_visualizer import RerunLogger, build_episode_reader
from delete import Deleter
from datetime import datetime
import logging_mp
import psutil

# shut down the logger in the rerun_visualizer
import rerun_visualizer as reader_mod
reader_mod.logger_mp.setLevel(logging_mp.ERROR)

if "XDG_RUNTIME_DIR" not in os.environ:
    runtime_dir = "/tmp/runtime-user"
    os.makedirs(runtime_dir, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    os.environ["XDG_RUNTIME_DIR"] = runtime_dir

os.environ["RUST_LOG"] = "off"

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

def data_starisrics_report(data_number,valid_episode_number,empty_data_number,totaol_step_num):
    separator = "=" * 50
    if data_number != 0:
        # Calculate values
        success_rate = valid_episode_number / data_number
        avg_steps = totaol_step_num / data_number
        
        # Print the report
        logger_mp.info(separator)
        logger_mp.info(f"{'DATA STATISTICS REPORT':^50}") # Centered Title
        logger_mp.info(separator)
        logger_mp.info(f"Status:                 Rerun Finished")
        logger_mp.info(f"Total Data Count:       {data_number}")
        logger_mp.info(f"Valid Episodes:         {valid_episode_number}")
        logger_mp.info(f"Empty Data Count:       {empty_data_number}")
        logger_mp.info("-" * 50)  # Thinner internal separator
        logger_mp.info(f"Final Success Rate:     {success_rate:.3%}")
        logger_mp.info(f"Average Steps:          {avg_steps:.3f}")
        logger_mp.info(separator)
    
    else:
        logger_mp.info(separator)
        logger_mp.info(f"{'DATA STATISTICS REPORT':^50}")
        logger_mp.info(separator)
        logger_mp.info(f"Status:                 No Data Rerun")
        logger_mp.info(separator)


def _supports_directory_episodes(data_record_type):
    return str(data_record_type).lower() == "hirol"


def _episode_exists(data_record_type, data_dir, episode_data_number, shared_reader=None):
    if _supports_directory_episodes(data_record_type):
        episode_dir = f"episode_{str(episode_data_number).zfill(4)}"
        full_path = os.path.join(data_dir, episode_dir)
        return os.path.exists(full_path), full_path

    if shared_reader is None:
        return False, data_dir

    episode_count = getattr(shared_reader, "_episode_count", None)
    if episode_count is None:
        return True, data_dir
    exists = 0 <= int(episode_data_number) < int(episode_count)
    return exists, data_dir


def _maybe_delete_episode(data_record_type, episode_data_number, data_dir):
    if _supports_directory_episodes(data_record_type):
        logger_mp.info(f"deleting Episode {episode_data_number}")
        Deleter.delete_episodes(episode_data_number, data_dir)
        logger_mp.info(f"successfull delete Episode {episode_data_number}")
        return

    logger_mp.warning(
        f"Skip deleting episode {episode_data_number}: lerobotv3 datasets do not map to episode_* directories."
    )


if __name__ == "__main__":

    data_dir = "data/train_episode/moving_bread/moving_bread_hirol_lerobotv3"
    start_episode = 1
    end_episode = 30
    fps = 30
    skip_step_nums = 4
    # 根据电脑的运存大小调整运存上限 在终端里free -h查看avaliale内存大小 
    lim_storage = 20
    data_type = "human_hand" # robot & human_hand

    episode_list = range(start_episode, end_episode + 1)
    action_ori_type = "quaternion"
    umi_rotation_transform = {"right": [0.7071068, 0, 0.7071068, 0]}
    contain_ft = False
    data_record_type = "lerobotv3"
    shared_episode_reader = None
    if not _supports_directory_episodes(data_record_type):
        shared_episode_reader = build_episode_reader(
            task_dir=data_dir,
            data_record_type=data_record_type,
            action_type=ActionType.END_EFFECTOR_POSE,
            action_prediction_step=1,
            action_ori_type=action_ori_type,
            observation_type=ObservationType.END_EFFECTOR_POSE,
            rotation_transform=None,
            contain_ft=contain_ft,
            data_type=data_type,
        )
    mem_gb = 0
    valid_episode_number = 0 
    data_number = 0
    empty_data_number = 0
    totaol_step_num = 0

    for episode_data_number in episode_list:
        episode_dir = f"episode_{str(episode_data_number).zfill(4)}"
        logger_mp.info(f"episode_{str(episode_data_number).zfill(4)}")
        exists, full_path = _episode_exists(
            data_record_type,
            data_dir,
            episode_data_number,
            shared_reader=shared_episode_reader,
        )

        if exists:
            if shared_episode_reader is None:
                episode_reader = build_episode_reader(
                    task_dir=data_dir,
                    data_record_type=data_record_type,
                    action_type=ActionType.END_EFFECTOR_POSE,
                    action_prediction_step=1,
                    action_ori_type=action_ori_type,
                    observation_type=ObservationType.END_EFFECTOR_POSE,
                    rotation_transform=None,
                    contain_ft=contain_ft,
                    data_type=data_type,
                )
            else:
                episode_reader = shared_episode_reader
           
            logger_mp.info(f'find Episode {episode_data_number}')  
            episode_data = episode_reader.return_episode_data(episode_data_number, skip_step_nums)

            if episode_data is None or len(episode_data) == 0:
                logger_mp.warning(f'Episode {episode_data_number} has no data')
                # delete this data
                _maybe_delete_episode(data_record_type, episode_data_number, data_dir)
                valid_episode_number += 0
                empty_data_number += 1
                data_number += 0
                continue
            else:
                data_number += 1
                             
              # step counter
            step_num = skip_step_nums*len(episode_data) 
            totaol_step_num += step_num

            example_data = episode_data[0]
            state_keys = episode_reader._state_keys            
            online_logger = RerunLogger(
                task_dir=data_dir, 
                prefix=f"ep{episode_data_number}/", 
                IdxRangeBoundary=60, 
                memory_limit="20GB",
                example_item_data=example_data, 
                action_type=ActionType.END_EFFECTOR_POSE, 
                state_keys=state_keys
            )

            logger_mp.info(f'Episode {episode_data_number}:  {step_num} step')
            # logger_mp.info(f'{totaol_step_num}')

            # storage waring messages
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_gb += mem_info.rss / (1024 ** 3)
            if mem_gb <= lim_storage:
                 logger_mp.info(f"storage used: {mem_gb:.4f} GB")
            else:
                logger_mp.info(f"storage used: {mem_gb:.4f} GB")
                logger_mp.warning(f'storage used too much')
                logger_mp.warning(f'Close the rerun to release the storage')
                logger_mp.warning(f" Press 'c' to release storage")  
            
            # visuallize in fps
            for item_data in episode_data:
                online_logger.log_item_data(item_data)
                time.sleep(1/fps)

            logger_mp.info(f"Episode {episode_data_number} rerun finished。")  
            logger_mp.info(f" Press 'Enter' to continue ,'d' to delete this episode ,or 'ctrl c' to exit")
            user_input = input() 
            if user_input.lower() == 'd':
                _maybe_delete_episode(data_record_type, episode_data_number, data_dir)
                valid_episode_number += 0
                
            else:
                valid_episode_number += 1

            if mem_gb > lim_storage:
                mem_gb = 0
                os.system("pkill -x rerun")
                logger_mp.info(f"Rerun UI has closed, system has enough storage now")

        else:
            logger_mp.warning(f'Could not find episode {episode_data_number} in {full_path}')
            logger_mp.warning(f"Press 'Enter' to  continue, or 'ctrl c' to exit ")
            user_input = input() 

    if shared_episode_reader is not None and hasattr(shared_episode_reader, "close"):
        shared_episode_reader.close()

    data_starisrics_report(data_number,valid_episode_number,empty_data_number,totaol_step_num)
