# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0


def test_file_service_reexports_all_used_symbols():
    from services import file_service as fs

    for name in [
        "FileValidator",
        "FileParser",
        "save_uploaded_file",
        "delete_dataset_folder",
        "get_dataset_data",
        "get_jsonl_data",
        "get_parquet_data",
        "save_raw_parquet_with_mapping",
        "save_parquet_dataset_content",
        "save_dataset_content",
        "count_records",
        "stream_to_disk",
        "stream_split_jsonl",
        "stream_split_parquet",
        "remap_jsonl_file",
        "zip_folder",
        "get_training_file_path",
        "CONFIG",
        "upload_single_file",
        "upload_multiple_files",
        "list_uploaded_files",
        "delete_file",
    ]:
        assert hasattr(fs, name), f"file_service must re-export {name}"


def test_new_package_modules_importable():
    from services.file import parsing, reads, streaming, validation  # noqa: F401

    assert hasattr(validation, "FileValidator")
    assert hasattr(streaming, "stream_to_disk")
    assert hasattr(reads, "get_dataset_data")
    assert hasattr(parsing, "FileParser")
