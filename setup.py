# setup.py
import os
from setuptools import setup, find_packages


# Get version from __version__.py
cu_path = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(cu_path, 'robolearn_envs', '__version__.py'), 'r') as f:
    exec(f.read(), about)

# Get description from README
with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='robolearn_envs',
    version=about['__version__'],
    author="Domingo Esteban",
    author_email="mingo.esteban@gmail.com",
    description="A python package that contains some OpenAIGym-like robot "
                "environments in PyBullet.",
    long_description=long_description,
    packages=find_packages(),
    url="https://github.com/domingoesteban/robolearn_envs",
    install_requires=[
        'gym',
        'pybullet',
        'numpy',
        'matplotlib',
        'transforms3d',
        'sympy',  # Because transforms3d doesn't install it
    ],
    include_package_data=True,
)
