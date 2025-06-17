from setuptools import setup, find_packages

setup(
    name='FPT-Diffusion',
    version='0.1.0',
    author='martinj',
    author_email='martinj1@student.ubc.ca',
    description = "Free Particle Tracking (for) Diffusion. SPT, MPT, image processing, and other tools for research scientists around the world. ",
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
