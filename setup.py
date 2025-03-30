from setuptools import find_namespace_packages, setup

setup(
    name="plotflow",
    url="https://github.com/tboulet/PlotFlow-Aggregated-Metrics-Visualization",
    author="Timothé Boulet",
    author_email="timothe.boulet0@gmail.com",
    packages=find_namespace_packages(),
    version="1.0",
    license="MIT",
    description="Filtering, grouping, and visualizing metrics from a local folder, with many features.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "plotflow             = plotflow.plot:main",
            "plotflow-init-config = plotflow.init_config:main",
        ],
    },
)
