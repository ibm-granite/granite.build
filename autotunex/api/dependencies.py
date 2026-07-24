# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""
Centralized dependency injection functions for FastAPI routes
"""

from fastapi import Depends
from services import (
    config_service,
    dataset_service,
    db_service,
    gb_service,
    job_service,
    user_service,
)
from services.registry.base import ModelRegistry

_db_instance = None


def get_database() -> db_service.Database:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = db_service.Database()
    return _db_instance


_gb_instance = None


def get_gb_service() -> gb_service.GBService:
    """Get singleton GBService instance"""
    global _gb_instance
    if _gb_instance is None:
        _gb_instance = gb_service.GBService()
    return _gb_instance


def get_job_service(
    database: db_service.Database = Depends(get_database),
) -> job_service.Job:
    """Get Job service instance with database dependency"""
    return job_service.Job(database)


def get_config_service(
    database: db_service.Database = Depends(get_database),
) -> config_service.Config:
    """Get Config service instance with database dependency"""
    return config_service.Config(database)


def get_dataset_service(
    database: db_service.Database = Depends(get_database),
) -> dataset_service.Dataset:
    """Get Dataset service instance with database dependency"""
    return dataset_service.Dataset(database)


def get_dmf_service(
    database: db_service.Database = Depends(get_database),
) -> ModelRegistry:
    """Resolve the configured model-registry backend (dmf in IBM mode, local in OSS)."""
    from services.plugins import Seam, resolve

    return resolve(Seam.REGISTRY, db=database)


def get_user_service(
    database: db_service.Database = Depends(get_database),
) -> user_service.User:
    """Get User service instance with database dependency"""
    return user_service.User(database)
