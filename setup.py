from setuptools import setup, find_packages
from io import open

setup(
    name='cvmodels',
    packages=find_packages(where="."),
    version='0.0.1',
    description='Keras Implementations of Object Detection Models',
    author='Bibhash Pran Das',
    author_email='bibhashp.das@gmail.com',
    url='https://github.com/Bibhash12301/Optiver-realized-volatility-prediction-kaggle21/tree/main/src',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Computer Vision Models Tf/keras implementations",
        ]
)