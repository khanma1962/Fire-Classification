

   
from setuptools import setup


with open('README.md', 'r', encoding='utf-8') as f:
    long_decription = f.read()

setup(
    name='src',
    version='0.0.1',
    author='Moe Khan',
    description='A small package for DVC-DL-TF Pipline',
    long_decription=long_decription,
    long_decription_content_type='text/markdown',
    url="https://github.com/khanma1962/DVC_DL_TF_Demo",
    author_email='khan_m_a@hotmail.com',
    package=['src'],
    python_requires='>3.7',
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'dvc',
        'tensorflow',  
        'tqdm',
        'PyYAML',
        'boto3' 
    ]

)