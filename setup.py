from setuptools import setup


required_packages = [
    'torch>=1.8.0',
    'numpy==1.22.4',
    'pandas==2.0.0',
    'anndata>=0.7.6',
    'tqdm',
    'scikit-learn==1.2.2',
    'matplotlib',
    'seaborn'
]

setup(
    name = 'MODE-omics',
    version = '0.1.0',
    packages = ['MODE'],
    description = 'a multimodal autoencoder framework for high-resolution multi-omic digital dissociation',
    author = 'Jiao Sun',
    author_email = 'jiao.sun@stjude.org',
    url = 'https://github.com/jsuncompubio/MODE',
    license = 'GPL-3.0 License',
    python_requires = '>=3.10',
    platforms = 'any',
    install_requires = required_packages
)
