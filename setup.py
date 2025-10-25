from setuptools import setup

setup(
    name='BPETokenizer',
    version='0.1.0',
    description='BPE Tokenization Engine',
    url='https://github.com/AndrewWillis7/BPE-Tokenizer.git',
    author='Andrew Willis, Emily Kathryn',
    author_email='AndrewWillis771@outlook.com',
    license='BSD 2-clause',
    packages=['BPE_Tokenizer'],
    install_requires=['tqdm', 
                      'datasets',
                      'hf_xet setuptools'
                      ],
    
    classifiers=[
        'Development Status :: 1',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Windos :: Linux :: MacOS',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ]

)