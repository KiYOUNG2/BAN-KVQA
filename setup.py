import pathlib
from setuptools import find_packages, setup


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Long description
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# This call to setup() does all the work
setup(
    name="vqa",
    version="0.0.1",
    description="Bilinear Attention Networks for Visual Question Anwering",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KiYOUNG2/BAN-KVQA",
    author="eliza.dukim",
    author_email="eliza.dukim@gmail.com",
    license="GNU",

    packages=find_packages(where=[
        'bottom_up_attention_pytorch',
        'bottom_up_attention_pytorch/detectron2'
        ],
        ),
    install_requires=requirements,
)
