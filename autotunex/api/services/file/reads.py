# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Dataset reads (bounded previews), saves, listing, deletion, zip. Moved verbatim from file_service.py."""

import asyncio
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import pyarrow.parquet as pq
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .parsing import FileParser
from .streaming import (
    AUTOTUNE_DATASETS_PATH,
    CONFIG,
    stream_to_disk,
)
from .validation import FileValidator

# Module logger. Root logging is configured once at app startup
# (do not call basicConfig/setLevel here — see CLAUDE.md logging conventions).
logger = logging.getLogger(__name__)


async def save_uploaded_file(
    file: UploadFile,
    dataset_name: str,
    dataset_id: str = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Save an uploaded file to the server by streaming it to disk in bounded
    chunks (see ``stream_to_disk``), so memory stays flat for large uploads.
    Returns the saved file path.
    """
    try:
        # Resolve the destination path (unchanged layout vs. the previous impl).
        if timestamp is not None:
            save_path = os.path.join(
                CONFIG["UPLOAD_DIR"], dataset_name, f"{timestamp}_{file.filename}"
            )
        elif dataset_id is not None:
            save_path = os.path.join(
                CONFIG["UPLOAD_DIR"], dataset_id, dataset_name, f"{file.filename}"
            )
        else:
            save_path = os.path.join(
                CONFIG["UPLOAD_DIR"], dataset_name, f"{file.filename}"
            )

        # stream_to_disk creates the parent directory and copies in 1MB chunks.
        await stream_to_disk(file, save_path)

        logger.info(f"File saved successfully at: {save_path}")
        return save_path

    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")


### Methods ###
async def upload_single_file(
    file: UploadFile = File(...), description: Optional[str] = Form(None)
):
    """
    Upload and parse a single file (CSV, JSON, JSONL, or Text format)
    """
    try:
        # Validate file
        FileValidator.validate_file_size(file)
        file_format = FileValidator.validate_file_format(file.filename)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save file
        save_path = await save_uploaded_file(file, timestamp)

        # Parse content
        content = await FileParser.parse_file(file, file_format)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "format": file_format,
                "description": description,
                "content": content,
                "metadata": {
                    "content_type": file.content_type,
                    "timestamp": timestamp,
                    "save_path": save_path,
                    "record_count": len(content) if isinstance(content, list) else 1,
                },
            },
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}",
        )


async def upload_multiple_files(
    files: List[UploadFile] = File(...), description: Optional[str] = Form(None)
):
    """
    Upload and parse multiple files (CSV, JSON, JSONL, or Text format)
    """
    results = []
    errors = []

    for file in files:
        try:
            # Validate file
            FileValidator.validate_file_size(file)
            file_format = FileValidator.validate_file_format(file.filename)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save file
            save_path = await save_uploaded_file(file, timestamp)

            # Parse content
            content = await FileParser.parse_file(file, file_format)

            results.append(
                {
                    "filename": file.filename,
                    "format": file_format,
                    "content": content,
                    "metadata": {
                        "content_type": file.content_type,
                        "timestamp": timestamp,
                        "save_path": save_path,
                        "record_count": (
                            len(content) if isinstance(content, list) else 1
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            errors.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(
        status_code=200,
        content={
            "description": description,
            "successful_uploads": results,
            "failed_uploads": errors,
            "total_files": len(files),
            "successful_count": len(results),
            "failed_count": len(errors),
        },
    )


async def list_uploaded_files():
    """
    List all files in the upload directory
    """
    try:
        files = []
        for filename in os.listdir(CONFIG["UPLOAD_DIR"]):
            file_path = os.path.join(CONFIG["UPLOAD_DIR"], filename)
            if os.path.isfile(file_path):
                files.append(
                    {
                        "filename": filename,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "created": datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).isoformat(),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat(),
                    }
                )

        return {
            "upload_directory": CONFIG["UPLOAD_DIR"],
            "total_files": len(files),
            "files": files,
        }
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


async def delete_file(filename: str):
    """
    Delete a specific file from the upload directory
    """
    try:
        file_path = os.path.join(CONFIG["UPLOAD_DIR"], filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} not found")

        os.remove(file_path)
        return {"message": f"File {filename} deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


async def delete_dataset_folder(dataset_name: str):
    """
    Delete a specific folder from the upload directory
    """
    try:
        folder_path = os.path.join(CONFIG["UPLOAD_DIR"], dataset_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Folder {dataset_name} deleted successfully")
            return {"message": f"Folder {dataset_name} deleted successfully"}
        else:
            logger.info(f"Folder {dataset_name} does not exist")
            return {"message": f"Folder {dataset_name} does not exist"}
    except Exception as e:
        logger.error(f"Error deleting folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting folder: {str(e)}")


def get_training_file_path(dataset_name):
    """
    Attempts to find the training file in two possible locations:
    1. ./data_sets/{dataset_name}
    2. ./datasets

    Args:
        dataset_name (str): Name of the dataset

    Returns:
        str: Path to the training file if found

    Raises:
        FileNotFoundError: If neither path exists
    """
    # First path pattern
    path1 = os.path.join(AUTOTUNE_DATASETS_PATH, dataset_name)
    # Second path pattern
    path2 = os.path.join(".", "datasets")
    # Check first path
    if os.path.exists(path1):
        return path1

    # Check second path
    if os.path.exists(path2):
        return path2

    # If neither path exists, raise an error
    raise FileNotFoundError(
        f"Training file not found in either location:\n1. {path1}\n2. {path2}"
    )


def zip_folder(folder_path, output_zip=None, output_dir=None, overwrite=False):
    """
    Zip the contents of a folder into a ZIP file.

    Args:
        folder_path (str): Path to the folder to be zipped
        output_zip (str, optional): Name for the output ZIP file. If not provided,
                                   a name will be generated based on the folder name.
        output_dir (str, optional): Directory where the ZIP file should be saved.
                                   If not provided, the current directory is used.
        overwrite (bool): Whether to overwrite the output file if it already exists

    Returns:
        str: Path to the created or existing ZIP file
    """
    # Ensure folder path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist")

    # Create output filename if not provided
    if output_zip is None:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_zip = f"{folder_name}_{timestamp}.zip"

    # Ensure the output filename has .zip extension
    if not output_zip.endswith(".zip"):
        output_zip += ".zip"

    # Process output directory
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        # Combine output_dir and output_zip to get the full path
        full_output_path = os.path.join(output_dir, output_zip)
    else:
        # If no output_dir specified, use the filename as is
        # (which could be a relative or absolute path)
        full_output_path = output_zip

    # Get the absolute paths
    abs_folder = os.path.abspath(folder_path)
    abs_output = os.path.abspath(full_output_path)

    # Check if the zip file already exists
    if os.path.exists(abs_output):
        if not overwrite:
            logger.info(f"ZIP archive already exists: {abs_output}")
            return abs_output
        else:
            logger.info(f"Overwriting existing ZIP archive: {abs_output}")

    # Create the zip file
    logger.info(f"Creating ZIP archive: {abs_output}")
    with zipfile.ZipFile(abs_output, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the folder
        for root, dirs, files in os.walk(abs_folder):
            for file in files:
                # Get the absolute path of the file
                abs_file_path = os.path.join(root, file)
                # Get the relative path to include in the zip
                rel_path = os.path.relpath(abs_file_path, os.path.dirname(abs_folder))
                logger.info(f"Adding: {rel_path}")
                zipf.write(abs_file_path, rel_path)

    logger.info(f"ZIP archive created successfully: {abs_output}")
    return abs_output


async def get_jsonl_data(dataset, limit: Optional[int] = None):
    """
    Read a .jsonl file and return its records as a list of dicts.

    When ``limit`` is given, only the first ``limit`` records are read — the file
    is streamed line by line and reading stops early, so memory stays flat even
    for multi-GB datasets (previewing must not load the whole file into RAM).
    """
    FILE_PATH = os.path.join(CONFIG["UPLOAD_DIR"], dataset)
    if not os.path.exists(FILE_PATH):
        logger.error(f"File not found: {FILE_PATH}")
        raise HTTPException(status_code=404, detail="File not found")

    def _read():
        data = []
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and len(data) >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=500,  # file should be valid; server-side error
                        detail=f"Invalid JSON in file: {line.strip()}",
                    )
        return data

    return await asyncio.to_thread(_read)


async def get_parquet_data(dataset: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Read a parquet file and return its content as a list of dicts.

    When ``limit`` is given, only the first ``limit`` rows are materialized via
    ``iter_batches``, so previewing a large parquet file does not load the whole
    table into memory.
    """
    FILE_PATH = os.path.join(CONFIG["UPLOAD_DIR"], dataset)
    if not os.path.exists(FILE_PATH):
        logger.error(f"File not found: {FILE_PATH}")
        raise HTTPException(status_code=404, detail="File not found")

    def _read():
        if limit is None:
            return pq.read_table(FILE_PATH).to_pylist()
        rows: List[Dict] = []
        pf = pq.ParquetFile(FILE_PATH)
        try:
            for batch in pf.iter_batches(batch_size=max(1, limit)):
                rows.extend(batch.to_pylist())
                if len(rows) >= limit:
                    break
        finally:
            pf.close()
        return rows[:limit]

    return await asyncio.to_thread(_read)


async def get_dataset_data(
    dataset: str, data_format: str = "jsonl", limit: Optional[int] = None
) -> List[Dict]:
    """
    Read dataset file in the appropriate format, optionally limited to the first
    ``limit`` records (used for bounded previews).
    """
    if data_format == "parquet":
        return await get_parquet_data(dataset, limit=limit)
    else:
        return await get_jsonl_data(dataset, limit=limit)
