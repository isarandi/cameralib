from setuptools import setup

setup(
    name='cameralib',
    version='0.1.0',
    author='István Sárándi',
    author_email='sarandi@vision.rwth-aachen.de',
    packages=['cameralib'],
    scripts=[],
    license='LICENSE',
    description='Represent, manipulate and use camera calibration info in computer vision tasks',
    long_description='',
    python_requires='>=3.6',
    install_requires=[
        'transforms3d',
        'opencv-python',
        'numpy',
    ]
)
