from setuptools import setup, find_packages
setup(
    name='graph_neural_networks',
    version='0.1.0',
    packages=find_packages(include=['graph_neural_networks', 'graph_neural_networks.*']),
    install_requires=["torch", "torch_geometric", "networkx", "matplotlib", "numpy"],
)