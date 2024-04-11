from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='qfinance',
    version='0.1.0',
    author='Chen Jianjun',
    author_email='chen1554@e.ntu.edu.sg',
    packages=find_packages(),
    license='LICENSE.txt',
    description='option pricing with QAE',
    long_description=open('README.md').read(),
    install_requires=required
)
