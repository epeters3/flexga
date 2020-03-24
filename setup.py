from setuptools import setup, find_packages

setup(
    name="flexga",
    version="1.0.1",
    packages=find_packages(include=["flexga", "flexga.*"]),
    license="MIT",
    url="https://github.com/epeters3/flexga",
    author="Evan Peterson",
    author_email="evanpeterson17@gmail.com",
    description=(
        "A simple, multi-purpose genetic algorithm supporting both continuous "
        "and discrete design variables."
    ),
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    install_requires=["numpy>=1.18.1"],
)
