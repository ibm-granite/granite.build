"""
Local checksum cache manager for llmbsub input uploads.

Provides a cache of checksums indexed by path hash to avoid
recalculating checksums for unchanged directories.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import portalocker

logger = logging.getLogger(__name__)

# Default cache location (shared across all users)
DEFAULT_CACHE_BASE_PATH = "/proj/granite-build/llmb/upload"


class ChecksumCache:
    """Manages local checksum cache with mtime-based invalidation."""

    def __init__(self, cache_base_path: str = DEFAULT_CACHE_BASE_PATH):
        self.cache_dir = Path(cache_base_path) / "checksum_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "checksums.json"
        self._cache: Dict = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load checksum cache: {e}")
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk with file locking."""
        try:
            with portalocker.Lock(str(self.cache_file), "w", timeout=5) as f:
                json.dump(self._cache, f, indent=2)
        except portalocker.exceptions.LockException as e:
            logger.warning(f"Failed to acquire lock for checksum cache: {e}")
        except Exception as e:
            logger.warning(f"Failed to save checksum cache: {e}")

    def _get_cache_key(self, path: str) -> str:
        """Generate cache key from path."""
        abs_path = os.path.abspath(path)
        return hashlib.sha256(abs_path.encode()).hexdigest()[:16]

    def _get_dir_mtime(self, path: str) -> float:
        """Get the most recent mtime of any file in the directory."""
        max_mtime = 0.0
        path_obj = Path(path)

        if path_obj.is_file():
            return path_obj.stat().st_mtime

        for item in path_obj.rglob("*"):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                except (OSError, IOError):
                    # Skip files we can't access
                    continue

        return max_mtime

    def _get_all_mtimes(self, path: str) -> Dict[str, float]:
        """Get mtimes of all files in the directory for integrity checking."""
        mtimes = {}
        path_obj = Path(path)

        if path_obj.is_file():
            mtimes[str(path_obj)] = path_obj.stat().st_mtime
            return mtimes

        for item in path_obj.rglob("*"):
            if item.is_file():
                try:
                    mtimes[str(item)] = item.stat().st_mtime
                except (OSError, IOError):
                    continue

        return mtimes

    def get(self, path: str) -> Optional[str]:
        """Get cached checksum if still valid (mtime unchanged)."""
        key = self._get_cache_key(path)
        entry = self._cache.get(key)

        if not entry:
            return None

        # Check if mtime has changed
        current_mtime = self._get_dir_mtime(path)
        if entry.get("mtime") != current_mtime:
            logger.info(f"Checksum cache invalidated for {path} (mtime changed)")
            return None

        logger.debug(f"Cache hit for {path}: {entry.get('checksum')}")
        return entry.get("checksum")

    def set(self, path: str, checksum: str):
        """Store checksum in cache."""
        key = self._get_cache_key(path)
        self._cache[key] = {
            "path": os.path.abspath(path),
            "checksum": checksum,
            "mtime": self._get_dir_mtime(path),
            "cached_at": datetime.now().isoformat(),
        }
        self._save_cache()
        logger.debug(f"Cached checksum for {path}: {checksum}")

    def invalidate(self, path: str):
        """Remove entry from cache."""
        key = self._get_cache_key(path)
        if key in self._cache:
            del self._cache[key]
            self._save_cache()
            logger.debug(f"Invalidated cache for {path}")

    def get_all_mtimes(self, path: str) -> Dict[str, float]:
        """
        Public method to get all file mtimes for integrity monitoring.
        Used by upload service to detect modifications during upload.
        """
        return self._get_all_mtimes(path)
