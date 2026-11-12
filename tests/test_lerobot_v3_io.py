import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from diffusion_policy.common.lerobot_v3_io import LeRobotV3Dataset, LeRobotV3Writer


@pytest.mark.parametrize("singleton_vectors", [False, True], ids=["scalars", "singleton-vectors"])
def test_local_reader_preserves_parquet_singletons_and_scalar_columns(tmp_path, singleton_vectors):
    root = tmp_path / "dataset"
    writer = LeRobotV3Writer(
        str(root), fps=25,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,)},
            "observation.contact": {"dtype": "float32", "shape": (1,)},
            "observation.tau_ext_l1": {"dtype": "float32", "shape": (1,)},
        },
        video_keys=[],
    )
    for index in range(2):
        writer.add_frame({
            "observation.state": np.array([index, index + 1], dtype=np.float32),
            "observation.contact": np.array([index], dtype=np.float32),
            "observation.tau_ext_l1": np.array([index * 14], dtype=np.float32),
        })
    writer.save_episode("test")
    writer.finalize()

    parquet = next((root / "data").rglob("*.parquet"))
    if singleton_vectors:
        # Match clean-data derived fields: metadata shape (1,), stored as [1].
        table = pq.read_table(parquet)
        for key in ("observation.contact", "observation.tau_ext_l1"):
            values = [[value] for value in table.column(key).to_pylist()]
            table = table.set_column(
                table.schema.get_field_index(key), key,
                pa.array(values, type=pa.list_(pa.float32(), 1)),
            )
        pq.write_table(table, parquet)
    original_bytes = parquet.read_bytes()

    dataset = LeRobotV3Dataset(str(root))
    assert len(dataset) == 2
    contact = dataset.get_column("observation.contact")[1]
    torque = dataset.get_column("observation.tau_ext_l1")[1]
    assert contact.shape == torque.shape == (torch.Size([1]) if singleton_vectors else torch.Size([]))
    assert contact.item() == 1
    assert torque.item() == 14
    assert dataset.get_column("timestamp")[1].shape == torch.Size([])
    torch.testing.assert_close(dataset[1]["observation.state"], torch.tensor([1., 2.]))
    assert parquet.read_bytes() == original_bytes
