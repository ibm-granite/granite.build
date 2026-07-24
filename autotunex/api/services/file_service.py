# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible shim. The implementation now lives in the `file` package
(validation / parsing / streaming / reads). Existing imports
`from services import file_service` and `from .file_service import X` keep working.
Do not add logic here — edit the focused modules under services/file/."""

from .file import *  # noqa: F401,F403
from .file import (  # noqa: F401 -- explicit names for `from .file_service import (...)` consumers
    AUTOTUNE_DATASETS_PATH,
    CONFIG,
    STREAM_CHUNK_SIZE,
    FileParser,
    FileValidator,
    count_records,
    delete_dataset_folder,
    delete_file,
    get_dataset_data,
    get_jsonl_data,
    get_parquet_data,
    get_training_file_path,
    list_uploaded_files,
    remap_jsonl_file,
    save_dataset_content,
    save_parquet_dataset_content,
    save_raw_parquet_with_mapping,
    save_uploaded_file,
    stream_split_jsonl,
    stream_split_parquet,
    stream_to_disk,
    upload_multiple_files,
    upload_single_file,
    zip_folder,
)
