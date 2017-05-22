from setuptools import setup

import oview

def readme():
    with open("README.rst", "r") as f:
        return f.read()

setup(
    name="OView",
    version=oview.__version__,
    author=oview.__author__,
    description="multi-purpose omega hdf image and data viewer",
    long_description=readme(),
    packages=["oview"],
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Utilities",
    ],
    entry_points={
        'console_scripts': ['oview=oview:main'],
    },
)
