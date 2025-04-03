from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

__version__ = "0.0.0"
exec(open("tumor_profile/__init__.py").read())

setup(
    name="tumor_profile",
    version=__version__,
    packages=find_packages(),
    url="",
    license="",
    author="BostonGene",
    author_email="",
    description="",
    install_requires=install_requires,
    package_dir={"tumor_profile": "tumor_profile"},
    package_data={"tumor_profile": ["configs/*"]},
    dependency_links=["https://nexus.devbg.us/repository/pypi-all/simple"],
    include_package_data=True,
)
