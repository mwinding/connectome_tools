from setuptools import setup, find_packages

setup(name='contools',
      version='0.2',
      description='tools for analyzing connectomics data from CATMAID instances',
      url='http://github.com/mwinding/connectome_tools',
      author='Michael Winding',
      author_email='mwinding@alumni.nd.edu',
      license='MIT',
      packages=find_packages(include=['contools', 'contools.*']),
      install_requires=['tables', 'python-catmaid', 'graspy', 'anytree', 'upsetplot']
      )


