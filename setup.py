from setuptools import setup, find_packages

setup(
    name="fx-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "xgboost>=1.4.2",
        "plotly>=5.3.1",
        "cryptography>=3.4.7",
        "python-dotenv>=0.19.0",
        "joblib>=1.0.2",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.7b0",
            "flake8>=3.9.2",
            "isort>=5.9.3",
        ],
    },
    author="Tu Nombre",
    author_email="tu.email@empresa.com",
    description="Sistema de predicción de volúmenes FX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/fx-prediction",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
) 