# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""In-memory parsers + streaming record counter. Moved verbatim from file_service.py."""

from fastapi import UploadFile, HTTPException
import asyncio
import pandas as pd
import json
from typing import List
import io
import pyarrow.parquet as pq
import logging

# Module logger. Root logging is configured once at app startup
# (do not call basicConfig/setLevel here — see CLAUDE.md logging conventions).
logger = logging.getLogger(__name__)


class FileParser:
    @staticmethod
    async def parse_csv(file: UploadFile) -> List[dict]:
        content = await file.read()

        def _parse():
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("utf-8", errors="replace")
            return pd.read_csv(io.StringIO(text)).to_dict(orient="records")

        return await asyncio.to_thread(_parse)

    @staticmethod
    async def parse_json(file: UploadFile) -> dict:
        content = await file.read()

        def _parse():
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                text = content.decode("utf-8", errors="replace")
            return json.loads(text)

        return await asyncio.to_thread(_parse)

    @staticmethod
    async def parse_jsonl(file: UploadFile) -> List[dict]:
        raw = await file.read()

        def _parse():
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")

            data = []
            for line in text.split("\n"):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise HTTPException(
                            status_code=400, detail=f"Invalid JSON line: {str(e)}"
                        )
            return data

        return await asyncio.to_thread(_parse)

    @staticmethod
    async def parse_parquet(file: UploadFile) -> List[dict]:
        content = await file.read()

        def _parse():
            table = pq.read_table(io.BytesIO(content))
            return table.to_pylist()

        return await asyncio.to_thread(_parse)

    @staticmethod
    async def parse_text(file: UploadFile) -> dict:
        """Parse text file content"""
        content = await file.read()

        def _parse():
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("utf-8", errors="replace")

            return {
                "content": text_content,
                "lines": text_content.split("\n"),
                "statistics": {
                    "total_lines": len(text_content.split("\n")),
                    "total_words": len(text_content.split()),
                    "total_characters": len(text_content),
                    "non_empty_lines": len(
                        [line for line in text_content.split("\n") if line.strip()]
                    ),
                },
            }

        return await asyncio.to_thread(_parse)

    @classmethod
    async def parse_file(cls, file: UploadFile, file_format: str) -> dict:
        parser_map = {
            "csv": cls.parse_csv,
            "json": cls.parse_json,
            "jsonl": cls.parse_jsonl,
            "parquet": cls.parse_parquet,
            "text": cls.parse_text,
        }

        try:
            parser = parser_map.get(file_format)
            if not parser:
                raise HTTPException(
                    status_code=400, detail=f"No parser found for format: {file_format}"
                )

            return await parser(file)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error parsing file content: {str(e)}"
            )


async def count_records(file_path: str, file_format: str) -> int:
    """
    Count records in a saved file without loading everything into memory.
    Runs in a thread to avoid blocking the event loop.
    """

    def _count():
        if file_format == "parquet":
            metadata = pq.read_metadata(file_path)
            return metadata.num_rows
        elif file_format == "jsonl":
            count = 0
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        elif file_format == "csv":
            # Count lines minus header
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1
        elif file_format == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return len(data) if isinstance(data, list) else 1
        else:
            return 0

    return await asyncio.to_thread(_count)
