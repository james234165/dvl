from setuptools import setup, find_packages
from os.path import join, curdir, exists

# extra requires
install_requires = [
    "numpy",
    "openai",
    "tqdm",
    "python-dotenv",
    "shortuuid",
    "anthropic",
    "pyyaml",
    "vllm",
    "datasets",
    "blobfile",
    "matplotlib",
    "levenshtein",
    "latex2sympy2"
]

setup(
    name="DVL Released Codes",
    version="0.1",
    description="",
    author="",
    author_email='',
    platforms=["linux-64", "osx-64"],
    license="Apache-2.0",
    url="",
    python_requires=">=3.10",
    install_requires=install_requires,
    packages=find_packages(),
)

