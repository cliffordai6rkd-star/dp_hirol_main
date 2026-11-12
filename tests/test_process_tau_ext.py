from pathlib import Path

import pytest
import yaml

from data_process.process_tau_ext import (
    DEFAULT_CONFIG, REPO_ROOT, build_export_args, load_config, validate_outputs,
)


def test_default_config_resolves_dataset_paths_from_repo_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(DEFAULT_CONFIG)
    assert config["datasets"][0]["input"] == str(REPO_ROOT / "data/huanggua1103_25hz_3cam_lerobotv3")
    assert config["datasets"][1]["output"] == str(REPO_ROOT / "data/board_25hz_3cam_lerobotv3_clean_ma5_l1_12p5")
    assert config["processing"]["urdf"] == str(REPO_ROOT / "scripts/assets/xarm7/xarm7_dynamics.urdf")


def test_yaml_controls_window_threshold_and_physics(tmp_path):
    pytest.importorskip("pinocchio")
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump({
        "datasets": [{"input": "input", "output": "output"}],
        "processing": {"tau_ext_smoothing_window": 9, "contact_threshold_l1": 11.25,
                       "cutoff_hz": 2.5, "payload_mass": 0.91},
    }))
    config = load_config(path, repo_root=tmp_path)
    args = build_export_args(config["datasets"][0], config["processing"])
    assert args.dataset == tmp_path / "input"
    assert args.output == tmp_path / "output"
    assert args.tau_ext_smoothing_window == 9
    assert args.contact_threshold_l1 == 11.25
    assert args.cutoff_hz == 2.5
    assert args.payload_mass == 0.91
    assert args.filter_mode == "causal"
    assert args.convention == "tool-on-environment"


@pytest.mark.parametrize("window", [0, -1, 2.5, True])
def test_window_must_be_positive_integer(window):
    pytest.importorskip("pinocchio")
    with pytest.raises(ValueError, match="positive integer"):
        build_export_args({"input": "input", "output": "output"}, {"tau_ext_smoothing_window": window})


def test_existing_output_is_rejected_before_any_conversion(tmp_path):
    source = tmp_path / "input"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError, match="choose a new output path"):
        validate_outputs([{"input": str(source), "output": str(output)}])


@pytest.mark.parametrize("overwrite", [False, True])
def test_pipeline_finishes_both_datasets_before_video(tmp_path, monkeypatch, overwrite):
    pytest.importorskip("pinocchio")
    import data_process.process_tau_ext as pipeline
    import data_process.tau_ext_video as video
    import scripts.add_xarm_wrench_v3 as exporter

    entries = []
    for name in ("a", "b"):
        source = tmp_path / name
        (source / "meta").mkdir(parents=True)
        (source / "meta/info.json").write_text("{}")
        entries.append({"input": str(source), "output": str(tmp_path / f"{name}_out")})
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump({"datasets": entries, "processing": {
        "tau_ext_smoothing_window": 7, "contact_threshold_l1": 12.5,
    }, "visualization": {"episodes": [2], "cameras": ["observation.images.front"]}}))
    events = []
    calls = []

    def export(args, *, overwrite=False):
        calls.append(overwrite)
        assert args.tau_ext_smoothing_window == 7
        assert args.contact_threshold_l1 == 12.5
        events.append(("data", args.output.name))
        (args.output / "meta").mkdir(parents=True)
        return args.output

    def render(root, episode, cameras, **kwargs):
        events.append(("video", root.name))
        assert episode == 2
        assert cameras == ["observation.images.front"]
        return {"video": root / "result.mp4"}

    monkeypatch.setattr(exporter, "export_dataset", export)
    monkeypatch.setattr(video, "export_tau_ext_video", render)
    outputs = pipeline.run_pipeline(config_path, overwrite=overwrite)
    assert calls == [overwrite, overwrite]
    assert [item[0] for item in events] == ["data", "data", "video", "video"]
    for output in outputs:
        assert (output / "meta/data_process_config.json").is_file()


def test_overwrite_accepts_existing_output_but_rejects_input_paths(tmp_path):
    source = tmp_path / "input"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    validate_outputs([{"input": str(source), "output": str(output)}], overwrite=True)
    for target in (source, tmp_path):
        with pytest.raises(ValueError, match="outside all input"):
            validate_outputs([{"input": str(source), "output": str(target)}], overwrite=True)
    other_source = tmp_path / "other_input"
    (other_source / "meta").mkdir(parents=True)
    (other_source / "meta/info.json").write_text("{}")
    with pytest.raises(ValueError, match="outside all input"):
        validate_outputs([
            {"input": str(source), "output": str(other_source)},
            {"input": str(other_source), "output": str(output)},
        ], overwrite=True)


def test_main_passes_overwrite_flag(monkeypatch):
    import data_process.process_tau_ext as pipeline

    calls = []
    monkeypatch.setattr(pipeline, "run_pipeline", lambda config, **kwargs: calls.append((config, kwargs)))
    pipeline.main(["--overwrite"])
    assert calls == [(DEFAULT_CONFIG, {"overwrite": True})]
