#!/usr/bin/env python3
"""
Setup script for yolo_model package
"""

from setuptools import setup, find_packages

setup(
    name="yolo_model",
    version="1.0.0",
    description="YOLO Model with Celery Integration",
    author="Benjidry",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "celery>=5.3.0",
        "redis>=4.5.0",
        "flower>=2.0.0",
        "opencv-python>=4.8.0",
        "ultralytics>=8.0.0",
        "numpy>=1.24.0",
        "Pillow>=9.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "yolo-celery=celery.celery_server:main",
            "yolo-client=celery.celery_client:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
