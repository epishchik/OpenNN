#!/usr/bin/env python
# pip install twine

import io
import os
import sys
from shutil import rmtree
from setuptools import setup, Command, find_packages

NAME = 'opennn_pytorch'
DESCRIPTION = 'Neural Networks library for image classification task.'
URL = 'https://github.com/Pe4enIks/OpenNN'
EMAIL = 'evgeniipishchik@mail.ru'
AUTHOR = 'Evgenii Pishchik'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = None

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirments.txt'), encoding='utf-8') as f:
    REQUIRED = f.read().split('\n')
print(REQUIRED)

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

GITHUB_URL = 'https://github.com/Pe4enIks/OpenNN'
DOWNLOAD_URL = GITHUB_URL + \
    f'/archive/refs/tags/v{about["__version__"]}.tar.gz'


class UploadCommand(Command):
    '''Support setup.py upload.'''

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        '''Prints things in bold.'''
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        build_str = '{0} '.format(sys.executable)
        build_str += 'setup.py sdist bdist_wheel --universal'
        os.system(build_str)

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    download_url=DOWNLOAD_URL,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    py_modules=[f'{NAME}'],
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    cmdclass={
        'upload': UploadCommand,
    },
)
