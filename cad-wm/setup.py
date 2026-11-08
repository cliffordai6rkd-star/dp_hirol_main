from setuptools import find_namespace_packages, setup


setup(
    name="cad-wm",
    version="0.1.0",
    description="CAD-WM dual-condition wrench flow and NEXT free-space torque estimation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10,<3.12",
    packages=find_namespace_packages(
        include=[
            "data_process*",
            "model*",
            "physics*",
            "scripts*",
            "train*",
        ]
    ),
    install_requires=[
        # These versions are the validated baseline of the pinn Conda
        # environment. NumPy 2.2 is required by Pinocchio 3.9/cmeel-boost.
        "numpy==2.2.6",
        "scipy==1.15.3",
        "PyYAML==6.0.3",
        "tqdm==4.70.0",
        "matplotlib==3.10.9",
        "h5py==3.16.0",
        "pandas==2.3.3",
        "pyarrow==19.0.1",
        "lerobot==0.4.0",
        "huggingface-hub[cli,hf-transfer]==0.35.3",
        "fsspec[http]==2025.3.0",
        "packaging==25.0",
        "wandb==0.21.4",
        "Pillow==12.2.0",
        # LeRobot and the server-side data tools use the headless OpenCV build.
        "opencv-python-headless==4.12.0.88",
        "rerun-sdk==0.26.2",
        # The CUDA wheel is installed from download.pytorch.org by setup.sh.
        # The versions also make a direct editable install reproducible.
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
        "torchcodec==0.2.1",
    ],
    extras_require={
        "physics": [
            "cmeel-tinyxml2==10.0.0",
            "cmeel-urdfdom==4.0.1",
            "libpinocchio==3.9.0",
            "pin==3.9.0",
            "mujoco==3.3.7",
        ],
        "cache": [
            # The default training cache is RAM. Zarr is only needed for the
            # optional train_data.cache.mode=ssd_zarr path.
            "zarr>=2.16,<3",
        ],
        "vision": [
            "open3d>=0.18,<1",
            "pyrealsense2",
            "ultralytics",
        ],
        "test": [
            "pytest==8.4.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "cad-wm-train=train.trainer.cad_wm_train:main",
            "next-train=train.trainer.tau_free_sequence_train:main",
            "cad-wm-build-offline-labels=data_process.tool.build_offline_tau_labels:main",
            "tau-other-train=train.trainer.tau_other_sequence_train:main",
        ]
    },
)
