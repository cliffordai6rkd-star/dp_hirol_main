import numpy as np

from data_process.tool.analyze_causal_rts_rnea_gap import summarize_gap


def test_summarize_gap_reports_per_joint_physical_statistics():
    gap = np.asarray(
        [
            [1.0, -2.0],
            [-1.0, 0.0],
        ]
    )

    summary = summarize_gap(gap, thresholds_nm=(0.5, 1.5), joint_names=("a", "b"))

    joint_a, joint_b = summary["per_joint"]
    assert joint_a["joint_name"] == "a"
    assert joint_a["bias_nm"] == 0.0
    assert joint_a["mae_nm"] == 1.0
    assert joint_a["rmse_nm"] == 1.0
    assert joint_a["fraction_above_threshold"] == {"0.5": 1.0, "1.5": 0.0}
    assert joint_b["bias_nm"] == -1.0
    assert joint_b["mae_nm"] == 1.0
    np.testing.assert_allclose(joint_b["rmse_nm"], np.sqrt(2.0))
    assert summary["overall"]["mae_nm"] == 1.0
