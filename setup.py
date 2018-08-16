from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='scikit-fem',
    version='0.1.8',
    description='Simple finite element assemblers',
    long_description='Easy to use finite element assemblers and related tools. See Github page for more information and examples.',  # Optional
    url='https://github.com/kinnala/scikit-fem',
    author='Tom Gustafsson',
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    keywords='sample setuptools development',  # Optional
    packages=find_packages(exclude=['examples']),  # Required
    install_requires=['numpy', 'scipy', 'sympy', 'matplotlib'],  # Optional
    extras_require={
        'all' : ['meshio'],
        },
)
