import argparse
import re
import shutil
from pathlib import Path

import h5py


def natural_sort_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rename teleop/tau_other to teleop/tau_other_cal in the last HDF5 files."
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("data/train_episode/nero_refinement"),
        help="Directory containing the HDF5 files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of files to select from the end after natural sorting.",
    )
    parser.add_argument("--source", default="teleop/tau_other")
    parser.add_argument("--target", default="teleop/tau_other_cal")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only show the selected files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a sibling .bak file before modifying each HDF5 file.",
    )
    return parser.parse_args()


def select_files(directory: Path, count: int):
    if count <= 0:
        raise ValueError("--count must be positive")
    if not directory.is_dir():
        raise FileNotFoundError(f"HDF5 directory does not exist: {directory}")

    files = sorted(
        [*directory.glob("*.h5"), *directory.glob("*.hdf5")],
        key=natural_sort_key,
    )
    if len(files) < count:
        raise ValueError(
            f"Found only {len(files)} HDF5 files in {directory}, "
            f"but --count is {count}."
        )
    return files[-count:]


def validate_file(path: Path, source_key: str, target_key: str):
    with h5py.File(path, "r") as h5_file:
        if source_key not in h5_file:
            raise KeyError(f"{path}: missing source dataset {source_key!r}")
        if target_key in h5_file:
            raise KeyError(f"{path}: target dataset already exists: {target_key!r}")

        source = h5_file[source_key]
        if not isinstance(source, h5py.Dataset):
            raise TypeError(f"{path}: source must be an HDF5 dataset")
        return source.shape, source.dtype


def replace_dataset(path: Path, source_key: str, target_key: str, backup: bool):
    if backup:
        backup_path = path.with_name(f"{path.name}.bak")
        if backup_path.exists():
            raise FileExistsError(
                f"Backup already exists: {backup_path}. Remove it or use --no-backup."
            )
        shutil.copy2(path, backup_path)

    with h5py.File(path, "r+") as h5_file:
        h5_file.move(source_key, target_key)
        h5_file.flush()


def main():
    args = parse_args()
    source_key = args.source.strip("/")
    target_key = args.target.strip("/")
    selected_files = select_files(args.directory, args.count)

    print("Selected HDF5 files:")
    for path in selected_files:
        shape, dtype = validate_file(path, source_key, target_key)
        print(f"  {path}  shape={shape} dtype={dtype}")

    if not args.apply:
        print("Dry run only. Add --apply to rename the datasets.")
        return

    if not args.no_backup:
        existing_backups = [
            path.with_name(f"{path.name}.bak")
            for path in selected_files
            if path.with_name(f"{path.name}.bak").exists()
        ]
        if existing_backups:
            raise FileExistsError(
                "Refusing to modify files because backups already exist: "
                + ", ".join(str(path) for path in existing_backups)
            )

    for path in selected_files:
        replace_dataset(path, source_key, target_key, backup=not args.no_backup)
        print(f"Renamed {source_key} -> {target_key}: {path}")


if __name__ == "__main__":
    main()
