"""Physical wrench sign/point conventions and a complete v3 metadata round trip."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pin = pytest.importorskip("pinocchio")
from scripts.add_xarm_wrench_v3 import (
    DEFAULT_URDF, FIELD_DIMS, WrenchEstimator, add_columns, export_dataset, parse_args,
)


def options(tmp_path, *extra):
    args = parse_args(["--dataset", str(tmp_path / "input"), "--output", str(tmp_path / "output"),
                       "--cutoff-hz", "0", "--damping", "1e-8", *extra])
    args.fps = 25.
    return args


def physical_sample(args, wrench):
    model = pin.buildModelFromUrdf(str(DEFAULT_URDF))
    data = model.createData()
    q = np.array([.2, -.3, .1, 1.2, .1, .7, .2])
    frame_id = model.getFrameId("link_eef")
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    jacobian = pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    gravity = pin.computeGeneralizedGravity(model, data, q).copy()
    # Known environment load enters dynamics with +J.T*w on the actuator side.
    torque = gravity - jacobian.T @ wrench
    return q, torque, jacobian


def test_known_load_sign_local_rotation_and_tcp_moment(tmp_path):
    args = options(tmp_path)
    w = np.array([2., -3., 4., .1, .2, -.3])
    q, torque, _ = physical_sample(args, w)
    t = np.arange(20) / 25
    state = np.tile(q, (len(t), 1))
    measured = np.tile(torque, (len(t), 1))
    out = WrenchEstimator(args).estimate_episode(state, np.zeros_like(state), measured, t)
    np.testing.assert_allclose(out["observation.wrench"], np.tile(w, (len(t), 1)), atol=2e-6)

    args.convention = "tool-on-environment"
    flipped = WrenchEstimator(args).estimate_episode(state, np.zeros_like(state), measured, t)
    np.testing.assert_allclose(flipped["observation.wrench"], -out["observation.wrench"], atol=2e-6)
    args.convention = "environment-on-tool"
    args.reference_frame = "local"
    args.tcp_offset = [.03, -.02, .11]
    estimator = WrenchEstimator(args)
    local = estimator.estimate_episode(state, np.zeros_like(state), measured, t)
    rotation = estimator.data.oMf[estimator.frame_id].rotation
    force_local = rotation.T @ w[:3]
    moment_local = rotation.T @ w[3:] - np.cross(args.tcp_offset, force_local)
    np.testing.assert_allclose(local["observation.wrench"][0],
                               np.r_[force_local, moment_local], atol=2e-6)


def test_payload_gravity_matches_independent_point_mass_jacobian(tmp_path):
    args = options(tmp_path, "--payload-mass", ".9", "--payload-com", ".01", "0", ".06")
    estimator = WrenchEstimator(args)
    q = np.array([.2, -.3, .1, 1.2, .1, .7, .2])
    bare = pin.buildModelFromUrdf(str(DEFAULT_URDF))
    data = bare.createData()
    pin.computeJointJacobians(bare, data, q)
    pin.updateFramePlacements(bare, data)
    fid = bare.getFrameId("link_eef")
    j = pin.getFrameJacobian(bare, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    offset_world = data.oMf[fid].rotation @ np.array(args.payload_com)
    gravity_force = args.payload_mass * np.array(args.gravity)
    load = np.r_[gravity_force, np.cross(offset_world, gravity_force)]
    expected = pin.computeGeneralizedGravity(bare, data, q).copy() - j.T @ load
    actual = pin.computeGeneralizedGravity(estimator.model, estimator.data, q)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("overwrite", [False, True])
def test_export_updates_data_hf_global_and_episode_stats_without_touching_source(tmp_path, monkeypatch, overwrite):
    args = options(tmp_path)
    root = args.dataset
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "videos/example.mp4").write_bytes(b"copied video fixture")
    n = 32
    times = np.arange(n, dtype=np.float32) / 25
    expected_wrenches = [np.array([1., 2., 3., .2, -.1, .3]), np.array([-5., 1., 2., -.2, .4, 0.])]
    q, tau1, _ = physical_sample(args, expected_wrenches[0])
    _, tau2, _ = physical_sample(args, expected_wrenches[1])
    original = pa.table({
        "timestamp": np.tile(times, 2), "index": np.arange(2*n),
        "episode_index": np.repeat([0, 1], n), "frame_index": np.tile(np.arange(n), 2),
    })
    original = add_columns(original, {
        "observation.joint": np.tile(q, (2*n, 1)), "observation.velocity": np.zeros((2*n, 7)),
        "observation.torque": np.repeat([tau1, tau2], n, axis=0),
    })
    # Include the HF schema and stale fingerprint normally present in LeRobot files.
    hf = {"info": {"features": {}}, "fingerprint": "old"}
    original = original.replace_schema_metadata({b"huggingface": json.dumps(hf).encode()})
    # One episode spans data files; filters/derivatives must be episode scoped.
    for i, chunk in enumerate([original.slice(0, 22), original.slice(22)]):
        pq.write_table(chunk, root / f"data/chunk-000/file-{i:03d}.parquet")
    pq.write_table(pa.table({"episode_index": [0, 1], "length": [n, n]}),
                   root / "meta/episodes/chunk-000/file-000.parquet")
    info = {"codebase_version": "v3.0", "fps": 25, "total_frames": 2*n,
            "total_episodes": 2, "features": {}}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(json.dumps({"keep_existing": {"count": [2*n]}}))
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    if overwrite:
        args.output.mkdir()
        (args.output / "stale_video.mp4").write_bytes(b"previous output")
    output = export_dataset(args, overwrite=overwrite)
    assert not (output / "stale_video.mp4").exists()
    after = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    result = pa.concat_tables([pq.read_table(p) for p in sorted((output / "data").rglob("*.parquet"))])
    w = np.asarray(result["observation.wrench"].to_pylist())
    expected = np.repeat(expected_wrenches, n, axis=0)
    np.testing.assert_allclose(w, expected, atol=3e-6)
    for key in original.column_names:
        assert result[key] == original[key]
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["keep_existing"] == {"count": [2*n]}
    for key in FIELD_DIMS:
        assert stats[key]["count"] == [2*n]
    np.testing.assert_allclose(stats["observation.wrench"]["mean"], expected.mean(0), atol=3e-6)
    episodes = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")
    assert episodes["stats/observation.wrench/count"].to_pylist() == [[n], [n]]
    np.testing.assert_allclose(episodes["stats/observation.wrench/mean"].to_pylist(), expected_wrenches, atol=3e-6)
    new_info = json.loads((output / "meta/info.json").read_text())
    assert new_info["features"]["observation.wrench"]["shape"] == [6]
    new_hf = json.loads(result.schema.metadata[b"huggingface"])
    assert new_hf["info"]["features"]["observation.wrench"]["length"] == 6
    assert "fingerprint" not in new_hf
    assert (output / "videos/example.mp4").read_bytes() == b"copied video fixture"
    assert (output / "visualizations/episode_000_tau_ext_l1.png").read_bytes().startswith(b"\x89PNG")
    if overwrite:
        import scripts.add_xarm_wrench_v3 as exporter

        saved = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}

        def fail_plot(*args, **kwargs):
            raise RuntimeError("plot failed")

        monkeypatch.setattr(exporter, "plot_episode", fail_plot)
        with pytest.raises(RuntimeError, match="plot failed"):
            export_dataset(args, overwrite=True)
        assert saved == {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_filter_resets_at_episode_boundary_and_keeps_static_gravity_zero(tmp_path):
    args = options(tmp_path, "--cutoff-hz", "3")
    estimator = WrenchEstimator(args)
    q, tau, _ = physical_sample(args, np.zeros(6))
    t = np.arange(30) / 25
    state = np.tile(q, (len(t), 1))
    first = estimator.estimate_episode(state, np.zeros_like(state), np.tile(tau + 10, (len(t), 1)), t)
    second = estimator.estimate_episode(state, np.zeros_like(state), np.tile(tau, (len(t), 1)), t)
    assert np.linalg.norm(first["observation.wrench"]) > 1
    np.testing.assert_allclose(second["observation.wrench"], 0, atol=1e-10)
    with pytest.raises(ValueError, match="monotonic"):
        estimator.estimate_episode(state, np.zeros_like(state), np.tile(tau, (len(t), 1)), t[::-1])


def test_threshold_is_joint_vector_l1_and_equality_is_retained():
    from diffusion_policy.common.xarm_tau_ext import gate_tau_ext
    # No single joint reaches 14, but the total does: keep every signed component.
    raw = np.array([3, -3, 2, -2, 2, -1, 1], np.float32)
    gated, l1, contact = gate_tau_ext(raw, 14)
    assert l1 == 14 and contact
    np.testing.assert_array_equal(gated, raw)
    gated, l1, contact = gate_tau_ext(raw * .9, 14)
    assert l1 < 14 and not contact
    np.testing.assert_array_equal(gated, np.zeros(7))


@pytest.mark.parametrize("window", [1, 5])
def test_causal_export_matches_online_and_future_does_not_change_prefix(tmp_path, window):
    from diffusion_policy.common.xarm_tau_ext import CausalTauExt, TauExtConfig
    args = options(tmp_path, "--filter-mode", "causal", "--contact-threshold-l1", "14",
                   "--convention", "tool-on-environment", "--cutoff-hz", "3",
                   "--torque-bias", "1", "2", "3", "4", "5", "6", "7")
    args.tau_ext_smoothing_window = window
    rng = np.random.default_rng(12)
    n = 60
    q0, tau0, _ = physical_sample(args, np.zeros(6))
    q = np.tile(q0, (n, 1)) + rng.normal(0, .002, (n, 7))
    dq = rng.normal(0, .05, (n, 7))
    tau = np.tile(tau0, (n, 1)) + np.arange(7) + 1 + rng.normal(0, 5, (n, 7))
    tau[25:40] += 15
    t = np.arange(n) / 25
    offline = WrenchEstimator(args).estimate_episode(q, dq, tau, t)
    online = CausalTauExt(args.urdf, TauExtConfig.from_args(args))
    samples = [online.update(*x) for x in zip(q, dq, tau, t)]
    np.testing.assert_array_equal(offline["observation.tau_ext"], [s["tau_ext"] for s in samples])
    np.testing.assert_array_equal(offline["observation.tau_ext_unthresholded"],
                                  [s["tau_ext_unthresholded"] for s in samples])
    np.testing.assert_array_equal(samples[0]["ddq"], np.zeros(7))
    # Modify all future inputs: prior outputs must remain identical.
    future_tau = tau.copy()
    future_tau[30:] *= 100
    changed = WrenchEstimator(args).estimate_episode(q, dq, future_tau, t)
    np.testing.assert_array_equal(changed["observation.tau_ext"][:30], offline["observation.tau_ext"][:30])
    with pytest.raises(ValueError, match="sample rate"):
        online.update(q[-1], dq[-1], tau[-1], t[-1] + .2)
    online.reset()
    reset = online.update(q[0], dq[0], tau[0], 0)
    np.testing.assert_array_equal(reset["tau_ext"], samples[0]["tau_ext"])


def test_trailing_signed_mean_precedes_threshold_and_resets():
    from diffusion_policy.common.xarm_tau_ext import CausalTauExt, TauExtConfig
    config = TauExtConfig(cutoff_hz=0, compensation="none",
                          tau_ext_smoothing_window=5, contact_threshold_l1=12.5)
    processor = CausalTauExt(DEFAULT_URDF, config)
    # Cancellation must happen before taking absolute values; no future/zero padding.
    torques = np.zeros((9, 7), np.float32)
    torques[:, 0] = [12.5, -12.5, 30, 40, -5, 60, 70, 80, -100]
    torques[:, 1] = [0, 0, -2, -3, 4, -5, 6, -7, 8]
    samples = [processor.update(np.zeros(7), np.zeros(7), tau, i / 25)
               for i, tau in enumerate(torques)]
    expected = np.asarray([torques[max(0, i-4):i+1].mean(0, dtype=np.float64)
                           for i in range(len(torques))], np.float32)
    np.testing.assert_array_equal([s["tau_ext_unthresholded"] for s in samples], expected)
    assert samples[0]["contact"]  # Equality retained, first sample not divided by five.
    assert not samples[1]["contact"]  # Opposite signed vectors cancel.
    for sample, raw in zip(samples, expected):
        l1 = np.abs(raw.astype(np.float64)).sum()
        assert sample["tau_ext_l1"] == l1
        np.testing.assert_array_equal(sample["tau_ext"], raw if l1 >= 12.5 else np.zeros(7))
    processor.reset()
    reset = processor.update(np.zeros(7), np.zeros(7), torques[0], 0)
    np.testing.assert_array_equal(reset["tau_ext_unthresholded"], torques[0])


@pytest.mark.parametrize("window", [0, -1, 2.5, True])
def test_invalid_tau_ext_smoothing_window_is_rejected(window):
    from diffusion_policy.common.xarm_tau_ext import CausalTauExt, TauExtConfig
    with pytest.raises(ValueError, match="positive integer"):
        CausalTauExt(DEFAULT_URDF, TauExtConfig(tau_ext_smoothing_window=window))
