from setuptools import setup

package_name = 'pid_control'

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
    description='PID longitudinal and lateral controller for CARLA',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pid_controller_node = pid_control.pid_controller_node:main',
        ],
    },
)
