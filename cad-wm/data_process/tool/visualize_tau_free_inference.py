from pathlib import Path

from data_process.tool.sequence_torque_inference_visualizer import (
    TorqueVisualizationTask,
    build_parser,
    run_visualization,
)


TASK = TorqueVisualizationTask(
    name="tau_free",
    target_key="tau",
    target_label="measured free-space torque",
    prediction_label="tau_free prediction",
    default_output_dir=Path("outputs/inference_visualization/tau_free"),
    rollout_mode="tau_free",
)


def main() -> None:
    run_visualization(TASK, build_parser(TASK).parse_args())


if __name__ == "__main__":
    main()
