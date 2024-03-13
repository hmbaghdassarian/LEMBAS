# python setup.py develop
# python setup.py install
from setuptools import setup
from setuptools import find_packages


CLASSIFIERS = '''\
License :: OSI Approved :: MIT license
Programming Language :: Python :: 3.6 :: 3.11
Topic :: Genome-Scale Modeling
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
'''

DISTNAME = 'LEMBAS'
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
]

EXTRAS_REQUIRES = {'interactive': ['jupyter', 'ipykerne']
                  }

PACKAGES = [
    'LEMBAS'
]

with open('README.md') as f:
    long_description = f.read()

metadata = dict(
    name=DISTNAME,
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
    classifiers=[CLASSIFIERS],
    license=LICENSE
)


def setup_package() -> None:
    setup(**metadata)


if __name__ == '__main__':
    setup_package()