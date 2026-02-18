"""
Setup script for cervical cytology classifier
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = [
                line.strip()
                for line in f.readlines()
                if line.strip() and not line.startswith('#')
            ]
    return requirements

# Read long description
def read_long_description():
    """Read long description from README"""
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="cervical-cytology-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep learning system for cervical cytology image classification",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cervical-cytology-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cervical-train=scripts.train_model:main",
            "cervical-eval=scripts.evaluate_model:main",
            "cervical-predict=scripts.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.md"],
    },
    zip_safe=False,
)