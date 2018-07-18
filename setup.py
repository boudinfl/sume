# -*- coding: utf-8 -*-

# sume
# Copyright (C) 2014, 2015, 2018 Florian Boudin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Setup the sume package."""

import io
import os
import sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()


def _install_custom():
    def pip_install(package, cwd=None):
        subprocess.check_call([sys.executable,
                               '-m',
                               'pip',
                               'install',
                               package],
                              cwd=cwd)

    pip_install('numpy')
    pip_install('Cython')
    pip_install('wmd')
    pip_install('.', cwd='pybind11')
    pip_install('.', cwd='fastText')


class InstallCommand(install):
    """Hack to circumvent fasttext and wmd install limitations."""

    def run(self):
        """Hack to install wmd and fastText."""
        _install_custom()
        install.run(self)


class DevelopCommand(develop):
    """Hack to circumvent fasttext and wmd install limitations."""

    def run(self):
        """Hack to install wmd and fastText."""
        _install_custom()
        develop.run(self)


# Where the magic happens:
setup(
    name='sume',
    version='2.0.0',
    description='Automatic summarization library.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Florian Boudin',
    author_email='florian.boudin@univ-nantes.fr',
    url='https://github.com/boudinfl/sume',
    python_requires='>=3.4',
    packages=find_packages(exclude=('tests',)),
    setup_requires=['pytest-runner'],
    # wmd and fastText are installed in custom commands
    install_requires=['PuLP', 'gensim', 'nltk', 'sklearn'],
    tests_require=['pytest', 'pytest-datadir'],
    include_package_data=True,
    license='GNU General Public License v3 (GPLv3)',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing'
    ],
    cmdclass={
        'install': InstallCommand,
        'develop': DevelopCommand
    }
)
