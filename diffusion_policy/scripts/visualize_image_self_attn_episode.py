"""Visualize the image-only baseline's last DINOv3 self-attention layer.

Use the same CLI as visualize_cross_attention_episode.py with a checkpoint
from train_dp_dinobase_insert_usb. Average spatial patch queries and heads;
--attention-query last uses only the final spatial patch query (not CLS).
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
from diffusion_policy.scripts.self_attention_visualization import ImageSelfAttentionCapture


def parse_args(argv=None):
    return common.parse_args(
        argv,
        description="Render the image-only baseline's final DINOv3 patch self-attention.",
        output_prefix="image_self_attn",
        query_help=(
            "Average all spatial image-patch queries (all), or use the final "
            "spatial patch query (last, not CLS). Both average the attention heads."
        ),
    )


def run(args):
    return common.run(
        args,
        capture_factory=ImageSelfAttentionCapture,
        output_prefix="image_self_attn",
    )


if __name__ == "__main__":
    run(parse_args())
