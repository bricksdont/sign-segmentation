# This file is adapted from
# https://github.com/awslabs/sockeye/blob/master/setup.py

import os
import re
import subprocess
from setuptools import setup, find_packages
from contextlib import contextmanager

ROOT = os.path.dirname(__file__)


def get_long_description():
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
        markdown_txt = f.read()
        return markdown_txt


def get_version():
    version_re = re.compile(r'''__version__ = ['"]([0-9.]+)['"]''')
    init = open(os.path.join(ROOT, 'sign_language_segmentation', '__init__.py')).read()
    return version_re.search(init).group(1)


def get_git_hash():
    # noinspection PyBroadException
    try:
        sp = subprocess.Popen(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_str = sp.communicate()[0].decode("utf-8").strip()
        return out_str
    except:
        return "unkown"


@contextmanager
def temporarily_write_git_hash(git_hash, filename=os.path.join('sign_language_segmentation', 'git_version.py')):
    """Temporarily create a module git_version in sign_language_segmentation so that it will be
    included when installing and packaging."""
    content = """
# This file is automatically generated in setup.py
git_hash = "%s"
""" % git_hash
    if os.path.exists(filename):
        raise RuntimeError("%s already exists, will not overwrite" % filename)
    with open(filename, "w") as out:
        out.write(content)
    # noinspection PyBroadException
    try:
        yield
    except:
        raise
    finally:
        os.remove(filename)


requirements_map = {"git+https://github.com/sign-language-processing/datasets.git":
                        "sign-language-datasets @ git+https://github.com/sign-language-processing/datasets.git",
                    "git+https://github.com/bricksdont/pose-format.git@add_tf_tensor_tests":
                        "pose-format @ git+https://github.com/bricksdont/pose-format.git@add_tf_tensor_tests"}


def get_requirements(filename):
    with open(os.path.join(ROOT, filename)) as f:
        requirements =  []

        for line in f:
            line = line.rstrip()
            if "git+" in line:
                line = requirements_map[line]
            requirements.append(line)

        return requirements


install_requires = get_requirements('requirements.txt')

entry_points = {
    'console_scripts': [
        'sign-language-segmentation-create-tfrecord = sign_language_segmentation.create_tfrecord:main',
        'sign-language-segmentation-train = sign_language_segmentation.train:main',
    ],
}

args = dict(
    name='sign_language_segmentation',

    version=get_version(),

    description='Segmentation models for sign languages',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",

    url='https://github.com/bricksdont/sign-segmentation',

    author='Mathias Müller',
    author_email='mathias.mueller@uzh.ch',
    maintainer_email='mathias.mueller@uzh.ch',

    license='MIT License',

    python_requires='>=3',

    packages=find_packages(exclude=("test", "test.*")),

    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pillow'],

    install_requires=install_requires,

    entry_points=entry_points,

    package_data={
        # If any package contains *.poseheader files, include them:
        "": ["*.poseheader"],
    },

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
    ]
)

with temporarily_write_git_hash(get_git_hash()):
    setup(**args)
