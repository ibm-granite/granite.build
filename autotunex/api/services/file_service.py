# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible shim. The implementation now lives in the `file` package
(validation / parsing / streaming / reads). Existing imports
`from services import file_service` and `from .file_service import X` keep working.
Do not add logic here — edit the focused modules under services/file/."""

from .file import *  # noqa: F401,F403
from .file import (  # noqa: F401 -- explicit names for `from .file_service import (...)` consumers
    FileValidator,
    FileParser,
    count_records,
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
