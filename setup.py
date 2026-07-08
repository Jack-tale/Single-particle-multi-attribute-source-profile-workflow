"""Setup script for sp_source_profile package."""

from setuptools import setup, find_packages

setup(
    name="sp_source_profile",
    version="1.0.0",
    description="Single Particle Source Profile Builder — SEM-EDS based source profile construction with COBB uncertainty.",
    author="Peng",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "pandas>=1.3.0",
        "openpyxl>=3.0.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "parallel": ["joblib>=1.1.0"],
    },
    entry_points={
        "console_scripts": [
            "sp-build=sp_source_profile.cli:main",
        ],
    },
)
