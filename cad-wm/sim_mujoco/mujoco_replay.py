import argparse
import logging
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

if __package__:
    from .mujocosim_inteface import MujocoSim_interface_fr3
else:
    from mujocosim_inteface import MujocoSim_interface_fr3


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/sim_cfg/replay_test.yaml"


def _project_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class Sim_replayer:
    def __init__(self, config):
        if not isinstance(config, Mapping):
            raise TypeError("replay config must be a YAML mapping")
        self.config = dict(config)
        self._dataset = None

    def load_dataset(self):
        if self._dataset is not None:
            return self._dataset

        repo_id = self.config.get("repo_id")
        if not repo_id:
            raise ValueError("missing required replay config value: repo_id")

        root_value = self.config.get("root")
        if not root_value:
            raise ValueError("missing required replay config value: root")
        root = _project_path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset root does not exist: {root}")

        log.info("loading LeRobot dataset from %s", root)
        self._dataset = LeRobotDataset(
            repo_id=str(repo_id),
            root=root,
            video_backend=self.config.get("video_backend", "torchcodec"),
        )
        return self._dataset

    @staticmethod
    def _dataset_fps(dataset):
        try:
            fps = float(dataset.fps)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("dataset metadata does not contain a valid fps") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"dataset fps must be positive and finite, got {fps}")
        return fps

    def load_q_sequence(self, ep_idx, expected_q_size):
        dataset = self.load_dataset()
        total_episodes = int(dataset.num_episodes)
        if ep_idx < 0 or ep_idx >= total_episodes:
            raise IndexError(
                f"episode index {ep_idx} is outside dataset range 0..{total_episodes - 1}"
            )

        q_key = self.config.get("q_key", "observation.joint")
        episode = dataset.meta.episodes[ep_idx]
        start_frame_idx = int(episode["dataset_from_index"])
        end_frame_idx = int(episode["dataset_to_index"])
        if end_frame_idx <= start_frame_idx:
            raise ValueError(f"episode {ep_idx + 1} contains no frames")

        q_seq = []
        for frame_idx in range(start_frame_idx, end_frame_idx):
            frame = dataset.hf_dataset[frame_idx]
            if q_key not in frame:
                raise KeyError(f"joint feature {q_key!r} is missing from dataset frame")

            q = frame[q_key]
            if hasattr(q, "detach"):
                q = q.detach().cpu().numpy()
            q = np.asarray(q, dtype=np.float64)
            if q.shape != (expected_q_size,):
                raise ValueError(
                    f"frame {frame_idx} feature {q_key!r} has shape {q.shape}; "
                    f"expected ({expected_q_size},)"
                )
            if not np.all(np.isfinite(q)):
                raise ValueError(f"frame {frame_idx} feature {q_key!r} contains NaN or inf")
            q_seq.append(q)

        return q_seq

    def _episode_range(self, dataset):
        total_episodes = int(dataset.num_episodes)
        if total_episodes <= 0:
            raise ValueError("dataset contains no episodes")

        start = int(self.config.get("episode_start", 1))
        end = int(self.config.get("episode_end", min(10, total_episodes)))
        if start < 1 or end < start or end > total_episodes:
            raise ValueError(
                "episode_start and episode_end are 1-based and inclusive; "
                f"got {start}..{end} for a dataset with {total_episodes} episodes"
            )
        return range(start - 1, end)

    def replay(self):
        dataset = self.load_dataset()
        fps = self._dataset_fps(dataset)
        episode_indices = self._episode_range(dataset)

        viewer = MujocoSim_interface_fr3(self.config)
        viewer.load_model()
        viewer.print_model_info()
        viewer.save_compiled_mjcf()

        all_q_seqs = []
        log.info(
            "replaying dataset episodes %d through %d at %.3f fps",
            episode_indices.start + 1,
            episode_indices.stop,
            fps,
        )
        for ep_idx in episode_indices:
            log.info("loading episode %d", ep_idx + 1)
            q_seq = self.load_q_sequence(ep_idx, len(viewer.arm_joint_names))
            log.info("loaded episode %d with %d frames", ep_idx + 1, len(q_seq))
            all_q_seqs.append(q_seq)

        viewer.play_joint_sequences(all_q_seqs, dt=1.0 / fps)

    def replayer(self, config=None):
        """Compatibility wrapper for callers using the original method name."""
        if config is not None:
            if not isinstance(config, Mapping):
                raise TypeError("replay config must be a YAML mapping")
            self.config = dict(config)
            self._dataset = None
        return self.replay()


def parse_args():
    parser = argparse.ArgumentParser(description="Replay LeRobot joint data in MuJoCo")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--episode-start", type=int, help="first episode, 1-based")
    parser.add_argument("--episode-end", type=int, help="last episode, 1-based and inclusive")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the complete replay without opening the MuJoCo viewer",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="do not wait for Enter between episodes",
    )
    return parser.parse_args()


def load_config(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"replay config does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError(f"replay config must contain a YAML mapping: {path}")
    return dict(config)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.episode_start is not None:
        config["episode_start"] = args.episode_start
    if args.episode_end is not None:
        config["episode_end"] = args.episode_end
    if args.headless:
        config["show_sim"] = False
    if args.no_pause:
        config["pause_between_episodes"] = False

    Sim_replayer(config).replay()


if __name__ == "__main__":
    main()
