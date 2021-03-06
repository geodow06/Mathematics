import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="godmathlib", # Replace with your own username
    version="0.0.1",
    author="George Downe",
    author_email="georgedowne06@gmail.com",
    description="A math library made for my personal use case",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geodow06/Mathematics",
    packages=setuptools.find_packages(),
    install_requires=['numpy','time'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='~=3.8.2',
)