# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""`file` package — focused split of the former file_service.py.
Re-exports the full public surface so `from services.file import X` works."""

from .parsing import FileParser, count_records  # noqa: F401
from .reads import (  # noqa: F401
    delete_dataset_folder,
    delete_file,
    get_dataset_data,
    get_jsonl_data,
    get_parquet_data,
    get_training_file_path,
    list_uploaded_files,
    save_uploaded_file,
    upload_multiple_files,
    upload_single_file,
    zip_folder,
)
from .streaming import (  # noqa: F401
    AUTOTUNE_DATASETS_PATH,
    CONFIG,
    STREAM_CHUNK_SIZE,
    remap_jsonl_file,
    save_dataset_content,
    save_parquet_dataset_content,
    save_raw_parquet_with_mapping,
    stream_split_jsonl,
    stream_split_parquet,
    stream_to_disk,
)
from .validation import FileValidator  # noqa: F401
