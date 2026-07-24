# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""File-format / size validation. Moved verbatim from file_service.py."""

from fastapi import UploadFile, HTTPException
import logging

from .streaming import CONFIG

# Module logger. Root logging is configured once at app startup
# (do not call basicConfig/setLevel here — see CLAUDE.md logging conventions).
logger = logging.getLogger(__name__)


class FileValidator:
    @staticmethod
    def validate_file_size(file: UploadFile) -> bool:
        # Measure size without disturbing the caller's read position: validation
        # should not have the side effect of rewinding the stream.
        original_pos = file.file.tell()
        file.file.seek(0, 2)  # Seek to end of file
        file_size = file.file.tell()  # Current position == file size
        file.file.seek(original_pos)  # Restore original position

        if file_size > CONFIG["MAX_FILE_SIZE"]:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {CONFIG['MAX_FILE_SIZE'] // (1024 * 1024)}MB",
            )
        return True

    @staticmethod
    def validate_file_format(filename: str) -> str:
        extension = "." + filename.split(".")[-1].lower()

        for format_type, format_info in CONFIG["SUPPORTED_FORMATS"].items():
            if extension in format_info["extensions"]:
                return format_type

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(CONFIG['SUPPORTED_FORMATS'].keys())}",
        )
