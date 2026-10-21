"""Checkpoint-general entry point for episode attention visualization."""

if __name__ == "__main__":
    import os
    import pathlib
    import sys

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from diffusion_policy.scripts.visualize_image_query_cross_attention_episode import parse_args, run


if __name__ == "__main__":
    run(parse_args())
