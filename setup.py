import setuptools

setuptools.setup(
    name='markovattribution',
    version='0.0.1',
    author='Kailin L',
    description='Fit a Markov attribution model',
    url='https://github.com/kailin-lu/markovattribution.git',
    packages=setuptools.find_packages(),
    install_requires=[
        'scipy',
        'pandas',
        'numpy',
        'sklearn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)