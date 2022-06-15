import sys
from setuptools import setup, find_packages


assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "Bullet-Safety-Gym uses Python 3.6 and above. "

with open('README.md', 'r') as f:
    # description from readme file
    long_description = f.read()


setup(
    name='bullet_safety_gym',
    version='0.1',
    author='Sven Gronauer',
    author_email='sven.gronauer@tum.de',
    description='A framework to benchmark safety in Reinforcement Learning.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license='MIT license',
    url='https://github.com/svengronauer/Bullet-Safety-Gym',
    install_requires=[
        'gym>=0.17.2',
        'numpy',
        'pybullet>=3.0.6'
    ],
    python_requires='>=3.6',
    platforms=['Linux Ubuntu', 'darwin'],  # supports Linux and Mac OSX
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
