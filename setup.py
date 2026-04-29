from setuptools import find_packages, setup

setup(
    name="sog",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "finufft>=2.5.0",
        # "cufinufft>=2.5.0",
        "cufinufft @ git+https://github.com/Johnsonjz/finufft.git@feature/cufinufft-simple-plan-cache#subdirectory=python/cufinufft",
        "pytorch_finufft",
    ],
    author="SOG contributors",
    description="Gaussian long-range SOG plugin",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)
