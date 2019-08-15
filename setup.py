"""
WellLogs is a library that allows to process well data (logs, core photo etc.) and conveniently train
machine learning models.
"""
import re
from setuptools import setup, find_packages


with open('batchflow/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)


with open('docs/index.rst', 'r') as f:
    long_description = f.read()


setup(
    name='dataset',
    packages=find_packages(exclude=['examples']),
    version=version,
    url='https://github.com/analysiscenter/well_logs',
    license='Apache License 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for well data processing',
    long_description=long_description,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'tqdm==4.28.1',
        'numba==0.41.0',
        'matplotlib==3.0.2',
        'seaborn==0.9.0',
        'nbconvert==5.4.0',
        'lasio==0.23',
        'psutil==5.4.8',
        'numpy==1.15.4',
        'pytest==4.0.2',
        'feather_format==0.4.0',
        'multiprocess==0.70.7',
        'dill==0.2.9',
        'pandas==0.24.0',
        'setuptools==40.6.3',
        'scipy==1.1.0',
        'plotly==3.7.1',
        'scikit_image==0.14.1',
        'dask==1.0.0',
        'Pillow==6.1.0',
        'blosc==1.8.1',
        'feather==0.9.1dev',
        'skimage==0.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
    ],
)