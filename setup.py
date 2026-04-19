"""Setup script for Pulse"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pulse-observability",
    version="1.0.0",
    author="Pulse Team",
    author_email="team@pulse.dev",
    description="Distributed Tracing with ML Anomaly Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/Pulse",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Monitoring",
        "Topic :: Software Development :: Debuggers",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
    ],
    extras_require={
        "ml": [
            "scikit-learn>=1.3.0",
            "torch>=2.0.0",
        ],
        "storage": [
            "redis>=4.5.0",
            "elasticsearch>=8.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pulse-server=src.server:run_server",
        ],
    },
)
