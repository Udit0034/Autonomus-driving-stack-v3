from setuptools import setup

package_name = 'vehicle_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='dev@example.com',
    description='Lead vehicle detection using LiDAR for CARLA',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lead_vehicle_detector_node = vehicle_detection.lead_vehicle_detector_node:main',
        ],
    },
)
