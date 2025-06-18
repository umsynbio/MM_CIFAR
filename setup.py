from setuptools import setup, find_packages

setup(
    name="mmcifar",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "matplotlib"],
    author="Michigan Synthetic Biology Team",
    description="A dataset wrapper that performs CIFAR-10 multimodal loading",
    python_requires=">=3.7",
)
