import os
import json
import logging_mp
import glog as log
import numpy as np
import time

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

class Reader:
    def __init__(self, task_dir=".", json_file="data.json"):
        self.task_dir = task_dir
        self.json_file = json_file

    def get_episode_length(self, episode_idx: int) -> int:
        episode_dir = os.path.join(self.task_dir, f"episode_{episode_idx:04d}")
        json_path = os.path.join(episode_dir, self.json_file)

        if not os.path.exists(json_path):
            log.warn(f"Episode {episode_idx} data.json not found for {self.task_dir}.")
            return 0

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        data_list = json_data.get("data", [])
        if not data_list:
            return 0

        last_idx = data_list[-1].get("idx", None)
        if last_idx is None:
            return len(data_list)

        return int(last_idx) + 1


if __name__ == "__main__":
    data_dir = "dataset/data/0402_duo_unitree_pick_bread_87ep"
    start_episode = 1
    end_episode = 94
    start_time = time.perf_counter()
    episode_reader = Reader(task_dir=data_dir)

    valid_episode_count = 0
    total_step_num = 0
    empty_episodes = []
    missing_episodes = []

    for episode_idx in range(start_episode, end_episode + 1):
        episode_dirname = f"episode_{episode_idx:04d}"
        full_path = os.path.join(data_dir, episode_dirname)

        # logger_mp.info(episode_dirname)

        if not os.path.exists(full_path):
            missing_episodes.append(episode_idx)
            logger_mp.warning(f"missing: {episode_dirname}")
            continue

        length = episode_reader.get_episode_length(episode_idx)

        if length <= 0:
            empty_episodes.append(episode_idx)
            logger_mp.warning(f"Episode {episode_idx} has no data (length={length})")
            # Deleter.delete_episodes(episode_idx, data_dir)
            continue

        valid_episode_count += 1
        total_step_num += length
        # logger_mp.info(f"steps: {length}")

    read_time = time.perf_counter() - start_time
    logger_mp.info(f"valid episode count: {valid_episode_count}")
    logger_mp.info(f"read time: {read_time:.2f} s")
    logger_mp.info(f"average step: {total_step_num/valid_episode_count:.1f}")

    if empty_episodes:
        logger_mp.warning(f"empty episodes: {empty_episodes}")
    if missing_episodes:
        logger_mp.warning(f"missing episodes: {missing_episodes}")
