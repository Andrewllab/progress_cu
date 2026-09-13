from setuptools import find_namespace_packages, setup

setup(
    name="curation",
    version="0.0.1",
    packages=find_namespace_packages(include=["curation", "curation.*"]),
    python_requires=">=3.10",
)
