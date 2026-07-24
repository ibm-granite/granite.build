# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="autotunex-api-bridge",
    version="0.3.0",
    description="AutoTune Rest API for Logging",
    author="IBM Research",
    author_email="daniel.karl@ibm.com",
    license="Apache-2.0",
    python_requires=">=3.10,<4.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.11",
        "pydantic==2.11.10",
        "uvicorn==0.35.0",
        "pymysql==1.1.1",
        "DBUtils>=3.0",
        "python-dotenv==1.1.0",
        "email-validator==2.2.0",
        "python-multipart==0.0.20",
        "PyJWT>=2.8.0",
        "requests==2.32.2",
        "pytz==2026.1.post1",
    ],
)
