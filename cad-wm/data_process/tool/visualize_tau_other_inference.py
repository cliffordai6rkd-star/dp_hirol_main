from pathlib import Path

from data_process.tool.sequence_torque_inference_visualizer import (
    TorqueVisualizationTask,
    build_parser,
    run_visualization,
)


TASK = TorqueVisualizationTask(
    name="tau_other",
    target_key="tau_other",
    target_label="tau_other label",
    prediction_label="tau_other prediction",
    default_output_dir=Path("outputs/inference_visualization/tau_other"),
    rollout_mode="tau_other",
)


def main() -> None:
    run_visualization(TASK, build_parser(TASK).parse_args())


if __name__ == "__main__":
    main()
