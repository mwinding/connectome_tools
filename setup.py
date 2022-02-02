from setuptools import setup, find_packages

setup(name='contools',
      version='0.1',
      description='tools for analyzing connectomics data, focus on Drosophila larva dataset',
      url='http://github.com/mwinding/connectome_tools',
      author='Michael Winding and Ben Pedigo',
      author_email='mwinding@alumni.nd.edu',
      license='MIT',
      packages=find_packages(include=['contools', 'contools.*']),
      install_requires=['tables', 'python-catmaid', 'graspy']
      )


