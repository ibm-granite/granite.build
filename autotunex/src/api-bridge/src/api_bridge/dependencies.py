# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Centralized dependency injection for api-bridge FastAPI routes."""

from fastapi import Depends

from api_bridge import database
from api_bridge.services import (
    config_service,
    dataset_service,
    github_service,
    job_service,
    user_service,
)

_db_instance = None


def get_database() -> database.Database:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = database.Database()
    return _db_instance


def get_user_service(
    db: database.Database = Depends(get_database),
) -> user_service.User:
    """Get User service instance with database dependency."""
    return user_service.User(db)


def get_config_service(
    db: database.Database = Depends(get_database),
) -> config_service.Config:
    """Get Config service instance with database dependency."""
    return config_service.Config(db)


def get_dataset_service(
    db: database.Database = Depends(get_database),
) -> dataset_service.Dataset:
    """Get Dataset service instance with database dependency."""
    return dataset_service.Dataset(db)


def get_github_service() -> github_service.GitHubService:
    """Get GitHubService instance."""
    return github_service.GitHubService()


def get_job_service(
    db: database.Database = Depends(get_database),
) -> job_service.Job:
    """Get Job service instance with database dependency."""
    return job_service.Job(db)
