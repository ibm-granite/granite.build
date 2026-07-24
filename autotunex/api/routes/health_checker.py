# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from fastapi import FastAPI, HTTPException, status, APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone
import psutil
import httpx
from typing import Dict, Any
from services import db_service
from importlib.metadata import version

app = FastAPI()
database: db_service.Database = db_service.Database()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    uptime: float
    system_info: Dict[str, Any]
    dependencies: Dict[str, str]
    extras: Dict[str, str]


class HealthChecker:
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)

    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        }

    def get_uptime(self) -> float:
        """Calculate application uptime in seconds"""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()

    async def check_database(self) -> str:
        """Check database connection"""
        try:
            await database.test_db_connection_and_structure()
            return "healthy"
        except Exception as e:
            return f"unhealthy: {str(e)}"

    async def check_granite_build(self) -> str:
        """Check external API dependency"""
        try:
            async with httpx.AsyncClient() as client:
                return "healthy" #TODO: Implement a proper health check for the Granite Build API
                # return "healthy" if response.status_code == 401 else "unhealthy"
        except Exception as e:
            return f"unhealthy: {str(e)}"


health_checker = HealthChecker()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""

    # Check all dependencies
    dependencies = {
        "database": await health_checker.check_database(),
        "granite_build": await health_checker.check_granite_build(),
    }

    # Determine overall status
    overall_status = "healthy"
    for service, status_value in dependencies.items():
        if not status_value.startswith("healthy"):
            overall_status = "degraded"
            break

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        version=version("autotune-server"),
        uptime=health_checker.get_uptime(),
        system_info=health_checker.get_system_info(),
        dependencies=dependencies,
        extras={
            "autotune": version("autotune"),
            "torch": version("torch"),
            "gbcli": version("gbcli"),
            "dmf-lib": version("dmf-lib"),
        },
    )


@router.get("/health/live")
def liveness_probe():
    """Kubernetes liveness probe endpoint"""
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe endpoint"""
    try:
        # Check critical dependencies
        db_status = await health_checker.check_database()
        if not db_status.startswith("healthy"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready - database unavailable",
            )

        return {"status": "ready"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}",
        )
