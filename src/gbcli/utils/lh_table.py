"""Lh table module."""

import json
import re

import pandas as pd

from gbcli.utils.gbconstants import GB_DMF_LOADER_BATCH_SIZE, GB_DMF_USE_CLASSIC_LOADER

REGEX_VALID_ID_STR = r"^[a-z0-9][a-z0-9_]*$"
REGEX_VALID_ID = re.compile(REGEX_VALID_ID_STR)


# Function to validate an identifier
def is_valid_id(identifier: str) -> bool:
    # Define the regex pattern
    """Check if valid id."""
    return REGEX_VALID_ID.match(identifier) is not None


def createTableFromFile(
    lh,
    filepath,
    namespace,
    table_name,
    public,
):
    """Create table from file."""
    from lakehouse.assets.table import Table
    from lakehouse.core import TableDetails

    table_details = TableDetails(namespace=namespace, name=table_name, is_public=public)

    table = Table.from_filepath(
        lh=lh,
        table_details=table_details,
        filepath=filepath,
        use_batches=not GB_DMF_USE_CLASSIC_LOADER,
        batch_size=GB_DMF_LOADER_BATCH_SIZE,
    )

    return table


def createTableDataset(lh, df, namespace, table_name, type: str, public: bool):
    """Create table dataset."""
    try:

        from lakehouse.assets import Table  # type: ignore
        from lakehouse.assets.dataset import Dataset
        from lakehouse.core import TableDetails  # type: ignore
        from lakehouse.dataset_info import DatasetInfo, DatasetType
        from pyiceberg.exceptions import NoSuchTableError  # type: ignore

        if not is_valid_id(table_name):
            raise Exception(
                f"'{table_name}' is not a valid identifier for a Lakehouse table name. A valid identifier must match: {REGEX_VALID_ID_STR}"
            )

        try:
            lh.list_versions(f"{namespace}.{table_name}")
            raise Exception(
                f"Error: A Lakehouse table with the name '{table_name}' already exists. Please choose a different name."
            )
        except NoSuchTableError as e:

            if type == "table":
                # create table
                table_details = TableDetails(namespace=namespace, name=table_name, is_public=public)
                table = Table.from_dataframe(
                    lh=lh,
                    df=df,
                    table_details=table_details,
                )

                return table
            elif type == "dataset":
                dataset_info = DatasetInfo(
                    name=table_name,
                    type=DatasetType.REAL,
                    description=table_name,
                    is_public=public,
                )
                dataset_table_details = TableDetails(namespace=namespace, is_public=public)
                dataset = Dataset.from_dataframe(
                    lh=lh,
                    df=df,
                    dataset_info=dataset_info,
                    dataset_table_details=dataset_table_details,
                )
                return dataset

        except Exception as e:
            raise e
    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )

    except Exception as e:
        raise e


def hasNullValues(df: pd.DataFrame):
    """Check if null values."""
    return df.isnull().values.any()


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess df."""
    df.columns = df.columns.str.replace(r"[. ]", "_", regex=True).str.lower()
    for col in df.columns:
        if df[col].isnull().any():  # If column contains nulls
            if df[col].dtype == "object":  # String columns
                df[col] = df[col].fillna("")  # Replace with empty string
            elif df[col].dtype in ["int64", "float64"]:  # Numeric columns
                df[col] = df[col].fillna(0)  # Replace with 0
            else:  # Other types (e.g., datetime)
                df[col] = df[col].fillna(df[col].mode()[0])  # Use most common value
    return df


def convert_to_df(filepath: str, extension: str) -> pd.DataFrame:
    """Convert to df."""
    try:

        from lakehouse.assets.utils.dataset_utils import (  # type: ignore
            convert_to_dataframe,
        )

        if "jsonl" in extension:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                data = [json.loads(line) for line in f]
            return pd.json_normalize(data, sep="_")
        else:
            return convert_to_dataframe(filepath, extension)
    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )


def table_lh(lh, namespace: str, table_name: str):
    """Table lh."""
    try:

        from lakehouse.assets.table import Table  # type: ignore

        table = Table(
            lh=lh,
            namespace=namespace,
            table_name=table_name,
        )
        return table

    except ModuleNotFoundError:
        raise Exception(
            "Error: The dmf-lib package is required. Install it with 'pip install dmf-lib' or contact your administrator for access."
        )
    except Exception as e:
        if "Unauthorized" in str(e):
            raise Exception(
                f"Insufficient permissions to access the table {namespace}.{table_name}, or the table may not be present in the specified namespace."
            )
        else:
            raise e
