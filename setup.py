from setuptools import setup
import os

try:
    dependencies_managed_by_conda = os.environ['DEPENDENCIES_MANAGED_BY_CONDA'] == '1'
except KeyError:
    dependencies_managed_by_conda = False

setup(
    name='cameralib',
    version='0.1.3',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['cameralib'],
    scripts=[],
    license='LICENSE',
    description='Represent, manipulate and use camera calibration info in computer vision tasks',
    long_description='',
    python_requires='>=3.6',
    install_requires=[] if dependencies_managed_by_conda else [
        'opencv-python',
        'numpy',
        'scipy',
        'numba',
        'boxlib @ git+https://github.com/isarandi/boxlib.git',
    ]
)
