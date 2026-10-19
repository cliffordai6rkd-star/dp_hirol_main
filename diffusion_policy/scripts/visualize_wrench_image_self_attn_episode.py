"""Visualize wrench-to-image weights in the joint self-attention baseline.

Use the same CLI as visualize_cross_attention_episode.py with a checkpoint
from train_dp_selfattn_insert_usb. The final fusion layer's latest-observation
wrench queries are averaged over heads/queries and mapped to camera patches.
"""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from diffusion_policy.scripts import visualize_cross_attention_episode as common
from diffusion_policy.scripts.self_attention_visualization import WrenchImageSelfAttentionCapture


def parse_args(argv=None):
    return common.parse_args(
        argv,
        description="Render wrench-to-image attention in the joint self-attention baseline.",
        output_prefix="wrench_image_self_attn",
        query_help=(
            "Average all wrench queries at the latest observation (all), or use "
            "its last wrench query (last). Both average the attention heads."
        ),
    )


def run(args):
    return common.run(
        args,
        capture_factory=WrenchImageSelfAttentionCapture,
        output_prefix="wrench_image_self_attn",
    )


if __name__ == "__main__":
    run(parse_args())
