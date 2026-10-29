from setuptools import find_packages, setup

setup(
    name="sog",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"sog.tools": ["data/*.lmp"]},
    install_requires=[
        "numpy",
        "torch",
        "finufft>=2.5.0",
        "cufinufft>=2.5.0",
        # "cufinufft @ git+https://github.com/Johnsonjz/finufft.git@feature/cufinufft-simple-plan-cache#subdirectory=python/cufinufft",
        "pytorch_finufft",
    ],
    extras_require={
        # The isolated PyTorch-2.x deployment layer (sog.v2).
        # torch>=2.5 provides make_fx symbolic tracing + torch.library.register_autograd.
        "v2": ["torch>=2.5"],
        # AOTInductor .pt2 packaging (aoti_compile_and_package) landed after 2.5.
        "v2-aoti": ["torch>=2.6"],
    },
    author="SOG contributors",
    description="Gaussian long-range SOG plugin",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
