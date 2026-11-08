import argparse
from pathlib import Path
from typing import Any


class H5Dataset:
    def __init__(
        self,
        input_path: Path,
        max_episodes: int | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.max_episodes = max_episodes
        self.h5py = self.load_h5py()

    @staticmethod
    def load_h5py() -> Any:
        try:
            import h5py  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: h5py. Install it before reading H5 files.") from exc
        return h5py
    
    def files(self) -> list[Path]:
        if self.input_path.is_file():
            h5_files = [self.input_path]
        else:
            h5_files = sorted(self.input_path.glob("*.h5")) + sorted(self.input_path.glob("*.hdf5"))

        if not h5_files:
            raise FileNotFoundError(f"No .h5/.hdf5 files found under {self.input_path}")

        if self.max_episodes is not None:
            h5_files = h5_files[: self.max_episodes]
        return h5_files

    def _print_node(self, node: Any, prefix: str = "") -> None:
        for key in sorted(node.keys()):
            item = node[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, self.h5py.Dataset):
                print(f"{path}: dataset shape={item.shape} dtype={item.dtype}{self._format_attrs(item.attrs)}")
            elif isinstance(item, self.h5py.Group):
                print(f"{path}/: group{self._format_attrs(item.attrs)}")
                self._print_node(item, path)

    @staticmethod
    def _format_attrs(attrs: Any) -> str:
        if len(attrs) == 0:
            return ""
        parts = [f"{key}={attrs[key]!r}" for key in sorted(attrs.keys())]
        return " attrs={" + ", ".join(parts) + "}"

    def inspect(self) -> None:
        # 打印 H5 树结构
        for h5_path in self.files():
            print(f"\n# {h5_path}", flush=True)
            print("before open", flush=True)
            with self.h5py.File(h5_path, "r") as h5_file:
                # print("after open", flush=True)
                self._print_node(h5_file)
                # print("after print node", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the structure of H5/HDF5 episode files.")
    parser.add_argument(
        "--input_path",
        nargs="?",
        type=Path,
        default=Path("data/train_episode/wipe_board/wipe_board"),
        help="H5 file or directory containing .h5/.hdf5 files.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        help="Maximum number of H5 files to inspect. Use 0 for all files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_episodes = None if args.max_episodes == 0 else args.max_episodes
    reader = H5Dataset(args.input_path, max_episodes=max_episodes)
    reader.inspect()
