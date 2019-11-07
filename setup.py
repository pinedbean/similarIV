from setuptools import setup, find_packages

setup(
   name='similarIV',
   version='0.1',
   description='find similarity with information value and cosine similarity',
   author='Pined Laohapiengsak',
   author_email='pined@central.tech',
   packages=find_packages(),  #same as name
   install_requires=['numpy','pandas','scipy','matplotlib'], #external packages as dependencies
)
