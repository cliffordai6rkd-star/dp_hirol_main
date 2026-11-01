"""Rebuild a RenderNet training PNG from its local JSON metrics."""

import argparse
from pathlib import Path

from diffusion_policy.common.render_net_training_logger import MetricHistory, save_training_plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    history = MetricHistory.from_json(args.run_directory / "logs.json.txt")
    print(save_training_plot(history, args.run_directory / "training_metrics.png"))


if __name__ == "__main__":
    main()
