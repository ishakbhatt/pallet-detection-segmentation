from setuptools import setup
import os
from glob import glob

package_name = 'warehouse_segment_detect_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=['warehouse_segment_detect_publisher'],
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='isha',
    maintainer_email='ibhatt@umich.edu',
    description='Real-time ground segmentation and pallet detection',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference = warehouse_segment_detect_publisher.inference:main',
        ],
    },
)
