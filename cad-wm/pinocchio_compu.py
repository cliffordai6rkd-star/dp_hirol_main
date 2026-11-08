# pinocchio计算动力学线性项，映射成末端等效 wrench，并按 episode 画对比曲线。

import argparse
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml

from data_process.dataloader import PINNDataset


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare real with Pinocchio urdf tau.")
    parser.add_argument("--config", type=Path, default=Path("config/pinocchio.yaml"))
    return parser.parse_args()


def to_numpy_1d(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _save_episode_comparison(
    first,
    second,
    quantity_names,
    component_names,
    episode_index,
    save_path,
):
    difference = first - second
    curve_names = (
        quantity_names[0],
        quantity_names[1],
        f"{quantity_names[0]} - {quantity_names[1]}",
    )
    curve_values = (first, second, difference)
    colors = ("tab:blue", "#e6b800", "tab:green")

    fig, axes = plt.subplots(
        len(component_names),
        1,
        figsize=(14, max(2.4 * len(component_names), 4)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for component_idx, component_name in enumerate(component_names):
        axis = axes[component_idx]
        for curve_name, values, color in zip(curve_names, curve_values, colors):
            axis.plot(
                values[:, component_idx],
                label=curve_name,
                color=color,
                linewidth=1.1,
            )
        axis.set_ylabel(component_name)
        axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        axis.grid(True, alpha=0.35)
        axis.legend(loc="upper right", ncol=3)

    axes[-1].set_xlabel("frame in episode")

    fig.suptitle(f"Episode {episode_index}")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_two_quantities_by_episode(
    dataset: PINNDataset,
    calculate_quantities: Callable[[dict[str, Any]], tuple[Any, Any]],
    *,
    quantity_names: Sequence[str],
    component_names: Sequence[str],
    output_dir: Path,
    filename_suffix: str = "compare",
) -> list[Path]:
    """Read every episode, calculate two quantities, and save A/B/A-B plots."""
    quantity_names = tuple(quantity_names)
    component_names = tuple(component_names)
    if len(quantity_names) != 2:
        raise ValueError(
            f"quantity_names must contain exactly two names, got {len(quantity_names)}"
        )
    if not component_names:
        raise ValueError("component_names must not be empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_to_sample_idx = {
        raw_idx: sample_idx
        for sample_idx, raw_idx in enumerate(dataset.valid_indices)
    }
    episodes = list(dataset.dataset.meta.episodes)
    saved_paths = []
    log.info(f"num episodes: {len(episodes)}")

    for episode_position, episode in enumerate(episodes):
        episode_index = int(episode.get("episode_index", episode_position))
        start_raw = int(episode["dataset_from_index"])
        end_raw = int(episode["dataset_to_index"])
        first_values = []
        second_values = []

        for raw_idx in range(start_raw, end_raw):
            sample_idx = raw_to_sample_idx.get(raw_idx)
            if sample_idx is None:
                continue

            sample = dataset[sample_idx]
            first, second = calculate_quantities(sample)
            first = to_numpy_1d(first)
            second = to_numpy_1d(second)

            if first.shape != second.shape:
                raise ValueError(
                    f"episode {episode_index}, raw index {raw_idx}: "
                    f"quantity shapes do not match: {first.shape} != {second.shape}"
                )
            if first.size != len(component_names):
                raise ValueError(
                    f"episode {episode_index}, raw index {raw_idx}: got {first.size} components, "
                    f"but component_names contains {len(component_names)} names"
                )

            first_values.append(first)
            second_values.append(second)

        if not first_values:
            log.warning(f"episode {episode_index} has no samples, skip")
            continue

        first = np.stack(first_values, axis=0)
        second = np.stack(second_values, axis=0)
        save_path = output_dir / f"episode_{episode_index:03d}_{filename_suffix}.png"
        _save_episode_comparison(
            first,
            second,
            quantity_names,
            component_names,
            episode_index,
            save_path,
        )
        saved_paths.append(save_path)
        log.info(f"saved episode {episode_index} plot: {save_path}")

    return saved_paths


def main():
    args = parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pinocchio_config = config.get("pinocchio")
    plot_config = config.get("plot")
    if not isinstance(pinocchio_config, dict):
        raise ValueError("config must contain a 'pinocchio' mapping")
    if not isinstance(plot_config, dict):
        raise ValueError("config must contain a 'plot' mapping")

    try:
        urdf_path = Path(pinocchio_config["urdf_path"])
        frame_name = str(pinocchio_config["frame_name"])
        locked_joint_names = list(pinocchio_config["locked_joint_names"])
        output_dir = Path(plot_config["output_dir"])
    except KeyError as error:
        raise ValueError(f"missing required config key: {error.args[0]}") from error

    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    if not all(isinstance(name, str) and name for name in locked_joint_names):
        raise ValueError("pinocchio.locked_joint_names must be a list of joint names")

    dataset = PINNDataset(config)

    full_model = pin.buildModelFromUrdf(str(urdf_path))
    missing_joint_names = [
        name for name in locked_joint_names if not full_model.existJointName(name)
    ]
    if missing_joint_names:
        raise ValueError(f"joints not found in URDF: {missing_joint_names}")

    locked_joint_ids = [full_model.getJointId(name) for name in locked_joint_names]
    if locked_joint_ids:
        model = pin.buildReducedModel(
            full_model,
            locked_joint_ids,
            pin.neutral(full_model),
        )
    else:
        model = full_model
    data = model.createData()

    frame_id = model.getFrameId(frame_name)
    if frame_id == len(model.frames):
        raise ValueError(f"frame not found: {frame_name}")

    def calculate_wrenches(sample):
        q = to_numpy_1d(sample["q"][-1])
        v = to_numpy_1d(sample["v"][-1])
        a = to_numpy_1d(sample["a"][-1])
        tau = to_numpy_1d(sample["tau"][-1])
        # wrench = to_numpy_1d(sample["wrench"][-1])

        tau_id = pin.rnea(model, data, q, v, a)
        tau_g = pin.computeGeneralizedGravity(model, data, q).copy()
        # tau_id -= tau_g
        pin.computeJointJacobians(model, data, q)
        pin.framesForwardKinematics(model, data, q)
        jacobian = pin.getFrameJacobian(
            model,
            data,
            frame_id,
            pin.ReferenceFrame.LOCAL,
        )

        # tau_id_wrench = np.linalg.lstsq(jacobian.T, tau_id, rcond=None)[0]
        # tau_wrench = np.linalg.lstsq(jacobian.T, tau, rcond=None)[0]
        return tau, tau_id

    # plot_two_quantities_by_episode(
    #     dataset,
    #     calculate_wrenches,
    #     quantity_names=("ext_comu", "external sim in wrench space"),
    #     component_names=("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"),
    #     output_dir=output_dir,
    #     filename_suffix="wrench_compare",
    # )
    plot_two_quantities_by_episode(
        dataset,
        calculate_wrenches,
        quantity_names=("tau", "tau_id"),
        component_names=("1", "2", "3", "4", "5", "6", "7"),
        output_dir=output_dir,
        filename_suffix="tau_compare",
    )



if __name__ == "__main__":
    main()
