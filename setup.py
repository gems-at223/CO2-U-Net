from setuptools import setup, find_packages

setup(
    name="gems-at223",
    version="1.0.0",
    description="Attention based UNet for predicting gas saturation and pressure buildup in porous media",
    author="Anthony Tran",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
