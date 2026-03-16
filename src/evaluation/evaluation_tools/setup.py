from setuptools import setup

package_name = 'evaluation_tools'

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
    description='Evaluation logger and analysis tools for CARLA autonomy stack',
    license='MIT',
    entry_points={
        'console_scripts': [
            'evaluation_logger_node = evaluation_tools.evaluation_logger_node:main',
            'evaluate_localization = evaluation_tools.evaluate_localization:main',
            'evaluate_controller = evaluation_tools.evaluate_controller:main',
            'plot_trajectory = evaluation_tools.plot_trajectory:main',
            'generate_metrics_summary = evaluation_tools.generate_metrics_summary:main',
        ],
    },
)
