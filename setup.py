from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
exec(open(path.join(here, 'skfem/version.py')).read())

setup(
    name='scikit-fem',
    version=__version__,
    description='Simple finite element assemblers',
    long_description='Easy to use finite element assemblers and related tools. See Github page for more information and examples.',
    url='https://github.com/kinnala/scikit-fem',
    author='Tom Gustafsson',
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy', 'scipy', 'meshio>=4.0.4'],
    extras_require={
        'full': ['matplotlib'],
    },
    test_suite='tests',
)
