# python setup.py develop
# python setup.py install
from setuptools import setup
from setuptools import find_packages


classes = """
    License :: OSI Approved :: MIT License
    Programming Language :: Python :: 3
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Operating System :: Microsoft :: Windows
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

AUTHOR = 'Hratch Baghdassarian'
AUTHOR_EMAIL = 'hmbaghdassarian@gmail.com'
DESCRIPTION = 'Re-implementation of LEMBAS (https://github.com/Lauffenburger-Lab/LEMBAS)'
LICENSE = 'MIT'

VERSION = '0.1.0'
# ISRELEASED = False

# PYTHON_MIN_VERSION = '3.8'
# PYTHON_MAX_VERSION = '3.9'
# PYTHON_REQUIRES = f'>={PYTHON_MIN_VERSION}, <={PYTHON_MAX_VERSION}'

INSTALL_REQUIRES = [
    'pandas', # 1.4.0
    'scikit-learn', # 2.2.0'
    'plotnine', # 0.13.1
    'leidenalg', # 0.10.2
    'torch>=2.1.0',
    'annotated-types'
]

EXTRAS_REQUIRES = {'interactive': ['jupyter', 'ipykerne']
                  }

PACKAGES = [
    'LEMBAS'
]

with open('README.md') as f:
    long_description = f.read()

metadata = dict(
    name='LEMBAS_re',
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/hmbaghdassarian/LEMBAS',  # homepage
    packages=find_packages(include=('LEMBAS*'), exclude=('*test*',)),  # PACKAGES
    project_urls={'Documentation': 'https://hmbaghdassarian.github.io/LEMBAS/'},
    # python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    classifiers=classifiers,
    license=LICENSE
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == '__main__':
    setup_package()