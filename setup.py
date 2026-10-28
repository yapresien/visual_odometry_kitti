from setuptools import setup

setup(
    name='pykitti',
    version='0.3.1',
    description='A minimal gpu based demo for working with the KITTI dataset in Python',
    author='Anshul Yadav',
    author_email='er.anshulyadav@gmail.com',
    url='https://github.com/yapresien/visual_odometry_kitti.git',
    download_url='https://github.com/yapresien/visual_odometry_kitti.git',
    license='MIT',
    packages=['ay_pykitti_vo'],
    install_requires=['numpy', 'matplotlib', 'Pillow', 'pandas']
)
