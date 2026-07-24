# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

setup(
    name="autotune-server",
    version="1.7.0",
    description="AutoTune Rest API for Automated Fine Tuning",
    author="IBM Research",
    author_email="daniel.karl@ibm.com",
    license="Apache-2.0",
    python_requires=">=3.10,<4.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.11",
        "tuspyserver>=4.2.5",
        "pydantic==2.11.10",
        "uvicorn==0.35.0",
        "pymysql==1.1.1",
        "aiomysql>=0.2.0",
        "DBUtils>=3.1.0",
        "python-dotenv==1.1.0",
        "pytz>=2024.1",
        "psutil>=5.9.0",
        "pandas>=2.0",
        "pyarrow>=15.0",
        "requests==2.32.2",
        "email-validator==2.2.0",
        "python-multipart==0.0.20",
        "requests-oauthlib==2.0.0",
        "httpx==0.28.1",
        "PyJWT>=2.8.0",
        "fastmcp==3.1.1",
        "openai==2.29.0",
        "langchain==1.2.13",
        "langchain-openai==1.1.12",
        "langchain-mcp-adapters==0.2.2",
    ],
    extras_require={
        "granite-build": [
            "granite.build @ git+https://github.com/ibm-granite/granite.build.git",
            "autotune[core]",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
        ],
        "hf": [
            "huggingface_hub>=0.23",
        ],
    },
    entry_points={
        "autotunex.runners": [
            "local = services.runners.local_runner:LocalRunner",
            "gb = services.runners.gb_runner:GBRunner",
        ],
        "autotunex.model_registries": [
            "local = services.registry.local_backend:LocalRegistry",
            "dmf = services.registry.dmf_backend:DmfRegistry",
            "hf = services.registry.hf_backend:HuggingFaceRegistry",
        ],
        "autotunex.auth_providers": [
            "w3id = services.auth_providers.w3id_provider:W3idAuthProvider",
            "dev = services.auth_providers.dev_provider:DevAuthProvider",
        ],
        "autotunex.chat_providers": [
            "litellm = services.chat_providers.litellm_provider:LiteLLMChatProvider",
            "openai_compatible = services.chat_providers.openai_compat_provider:OpenAICompatChatProvider",
        ],
    },
)
