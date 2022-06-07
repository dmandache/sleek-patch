import setuptools
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(name='sleek',
    version='0.1',
    description='Texture aware patchification using SLIC superpixel algoithm',
    url='https://github.com/dmandache/sleek-patch',
    author='diana',
    install_requires=['scikit-image', 'numpy'],
    author_email='',
    packages=setuptools.find_packages(),
    long_description=long_description,
    zip_safe=False)
