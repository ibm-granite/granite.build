# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""`file` package — focused split of the former file_service.py.
Re-exports the full public surface so `from services.file import X` works."""

from .streaming import (  # noqa: F401
    STREAM_CHUNK_SIZE,
    AUTOTUNE_DATASETS_PATH,
    CONFIG,
    stream_to_disk,
    stream_split_jsonl,
    stream_split_parquet,
    remap_jsonl_file,
    save_dataset_content,
    save_raw_parquet_with_mapping,
    save_parquet_dataset_content,
)
from .validation import FileValidator  # noqa: F401
from .parsing import FileParser, count_records  # noqa: F401
from .reads import (  # noqa: F401
    save_uploaded_file,
    upload_single_file,
    upload_multiple_files,
    list_uploaded_files,
    delete_file,
    delete_dataset_folder,
    get_training_file_path,
    zip_folder,
    get_jsonl_data,
    get_parquet_data,
    get_dataset_data,
)
