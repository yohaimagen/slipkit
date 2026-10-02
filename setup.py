from setuptools import setup, find_packages

setup(
    name='slipkit',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['numpy>=1.24', 'scipy>=1.10', 'cutde>=25.7.24',
                      'meshio>=5', 'pandas>=2', 'matplotlib>=3.7'],
    extras_require={'bayesian': ['h5py>=3.12'], 'analysis': ['arviz>=0.20'],
                    'test': ['pytest>=8']},
)
