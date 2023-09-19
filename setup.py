from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

dependencies = [
    'tensorflow==2.13.*',
    'numpy',
    'opencv-python==4.5.*',
    'nvidia-cudnn-cu11==8.6.0.163',
    'statistics',
    'parameterized',
    'pdoc'
]

setup(
    name='basketballdetector',
    version='0.1dev',
    description='Basketball detection package',
    long_description=readme,
    author='Ivan Pelizon',
    author_email='ivan.pelizon@gmail.com',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'out', 'assets')),
    python_requires='>=3.8,<=3.11',
    install_requires=dependencies
)
