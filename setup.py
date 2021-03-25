"""Setup.py for HyperKA."""

import os

import setuptools

MODULE = 'hyperka'
VERSION = '1.0'
PACKAGES = setuptools.find_packages(where='src')
META_PATH = os.path.join('src', MODULE, '__init__.py')
KEYWORDS = ['Knowledge Graph', 'Hyperbolic Embeddings', 'Entity Alignment', 'Type Inference']
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'pandas',
    'psutil',
    'scikit-learn',
    'matplotlib',
    'python-igraph',
    'ray',
]

if __name__ == '__main__':
    setuptools.setup(
        name=MODULE,
        version=VERSION,
        description='Knowledge Association with Hyperbolic Knowledge Graph Embeddings',
        url='https://github.com/nju-websoft/HyperKA',
        author='Zequn Sun',
        author_email='zqsun.nju@gmail.com',
        maintainer='Zequn Sun',
        maintainer_email='zqsun.nju@gmail.com',
        license='MIT',
        keywords=KEYWORDS,
        packages=setuptools.find_packages(where='src'),
        package_dir={'': 'src'},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        zip_safe=False,
    )
