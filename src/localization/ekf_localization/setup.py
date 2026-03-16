from setuptools import setup

package_name = 'ekf_localization'

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
    description='EKF Localization node for CARLA sensor fusion',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ekf_localization_node = ekf_localization.ekf_localization_node:main',
        ],
    },
)
