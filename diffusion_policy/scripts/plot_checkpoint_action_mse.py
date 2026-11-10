"""Draw checkpoint action MSE; optionally refresh while a training PID runs."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot_scores(scores, output):
    records = sorted(
        (row for row in scores["records"] if math.isfinite(row["checkpoint_action_mse"])),
        key=lambda row: row["optimizer_step"],
    )
    if not records:
        return 0
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    steps = [row["optimizer_step"] / 1000 for row in records]
    values = [row["checkpoint_action_mse"] for row in records]
    minimum = min(range(len(records)), key=lambda index: values[index])
    protocol = records[-1].get("action_mse_protocol") or {}
    full_dataset = protocol.get("episode_selection") == "all"
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(steps, values, "o-", color="#2563eb", linewidth=2, label="EMA 动作 MSE")
    ax.scatter(steps[minimum], values[minimum], color="#dc2626", s=100, zorder=3,
               label=f"最低记录：{steps[minimum]:g}k / {values[minimum]:.6f}")
    visible = set(range(len(records))) if len(records) <= 12 else {0, len(records) - 1, minimum}
    for index in sorted(visible):
        row = records[index]
        episode_ids = row.get("episode_ids", [])
        episodes = (f"{len(episode_ids)} 条轨迹 / {row.get('num_windows', '?')} 窗口"
                    if len(episode_ids) > 5 else "episodes " + ",".join(map(str, episode_ids)))
        ax.annotate(f"{values[index]:.6f}\n{episodes}",
                    (steps[index], values[index]), xytext=(0, 14),
                    textcoords="offset points", ha="center", fontsize=9)
    title = "FDP 教师：全量动作 MSE" if full_dataset else "FDP 教师：checkpoint 动作 MSE"
    ax.set(title=title, xlabel="Optimizer steps（千步）",
           ylabel="Action MSE（原始动作单位）")
    ax.margins(x=0.12, y=0.40)
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    caption = ("全量训练轨迹，每个有效窗口评估一次，固定逐窗口推理噪声；这是训练集拟合误差。"
               if full_dataset else
               "每次随机抽取训练轨迹与推理噪声；不是固定验证集曲线，也不是训练去噪 loss。")
    fig.text(0.5, 0.035, caption,
             ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.png")
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    temporary.replace(output)
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--pid", type=int, help="Stop watching after this training process exits")
    parser.add_argument("--interval", type=float, default=15.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    source = args.run_dir / "checkpoints/action_mse.json"
    output = args.output or args.run_dir / "action_mse.png"
    previous = None
    while True:
        alive = True
        if args.pid is not None:
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError:
                alive = False
        # Read one final snapshot after the training process exits.
        if source.is_file():
            content = source.read_bytes()
            if content != previous:
                points = plot_scores(json.loads(content), output)
                previous = content
                print(f"{datetime.now().isoformat(timespec='seconds')} {points} points -> {output}", flush=True)
        if not args.watch or not alive:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
