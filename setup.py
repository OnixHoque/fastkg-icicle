from setuptools import setup, find_packages

setup(
    name='fastkg',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),  # Exclude test directories
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'ruamel.yaml'
    ],
    python_requires='>=3.8',
    author='Md Saidul Hoque Anik',
    author_email='mdshoque@iu.edu',
    description='An efficient sparse implementation of KG Translational Models',
)
