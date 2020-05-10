import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="actionrules-lukassykora",
    version="1.1.17",
    author="Lukas Sykora",
    author_email="lukassykora@seznam.cz",
    description="Action rules mining package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukassykora/actionrules",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'numpy',
        'pyfim'
    ],
)