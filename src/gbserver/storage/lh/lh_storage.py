from __future__ import annotations

import logging
import os
from multiprocessing import Lock
from typing import Callable, Optional, Type, Union

import pandas as pd
import pyarrow as pa
from pandas import DataFrame
from pydantic import BaseModel, Field, TypeAdapter
from tenacity import retry, stop_after_attempt, wait_exponential

from gbserver.storage.storage import UUID_COLUMN_NAME, BaseItemStorage, QueryControl
from gbserver.types.constants import (
    GRANITE_DOT_BUILD_ADMIN_NAMESPACE,
    LAKEHOUSE_BASE_DELAY,
    LAKEHOUSE_ENVIRONMENT,
    LAKEHOUSE_MAX_RETRIES,
    PYICEBERG_LOG_LEVEL,
)
from gbserver.utils.logger import LoggingUtility
from gbserver.utils.optional_imports import HAS_LAKEHOUSE

if HAS_LAKEHOUSE:
    from lakehouse import LakehouseIceberg
    from lakehouse.api import ConfigMap
    from lakehouse.assets.table import Table
    from lakehouse.core import PartitionColumn, TableDetails, TokenDetails
    from pyiceberg.expressions import And, BooleanExpression, EqualTo, IsNull, Or
    from pyiceberg.types import (
        BooleanType,
        FloatType,
        IntegerType,
        StringType,
        TimestampType,
        TimestamptzType,
    )

import datetime
from typing import Any, Self

from gbserver.utils.logger import get_logger

if HAS_LAKEHOUSE:
    pylogger = logging.getLogger("pyiceberg")
    pyi_log_level = logging.getLevelNamesMapping()[PYICEBERG_LOG_LEVEL]
    pylogger.setLevel(pyi_log_level)


logger = get_logger(__name__)

__lh_lock = Lock()
__lh_cache = {}
__default_config_yaml = "lh-conf.yaml"


def __cache_lh(key: Optional[str], lh: LakehouseIceberg) -> None:
    if key is None:
        key = __default_config_yaml
    global __lh_cache
    __lh_cache[key] = lh


def __get_cached_lh(key: Optional[str] = None) -> Optional[LakehouseIceberg]:
    if key is None:
        key = __default_config_yaml
    global __lh_cache
    return __lh_cache.get(key, None)


def log_lh_token_details(lh: LakehouseIceberg):
    details: TokenDetails = lh.get_token_details()
    if details.isExpiringInNextDays(7):
        logger.warning(
            f"LH Token is expiring in 7 days: email/user={details.email}, expiration={details.expiration}"
        )
    else:
        logger.info(
            f"LH Token details: email/user={details.email}, expiration={details.expiration}"
        )


from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg


def get_default_lh() -> LakehouseIceberg:
    """Get the singleton LakehouseIceberg instance used by default by BaseItemStorage.

    Raises:
        ValueError: If neither the env or lh-conf.yaml specified lh environment and token.

    Returns:
        LakehouseIceberg:
    """
    global __lh_lock
    error_str = "failed to create a Lakehouse client"
    with __lh_lock:
        lh = __get_cached_lh("map")
        if lh is not None:
            return lh
        lh = __get_cached_lh("env")
        if lh is not None:
            return lh
        token = os.environ.get("LAKEHOUSE_TOKEN", None)
        if token is not None:
            try:
                # By using LAKEHOUSE_ENVIRONMENT, we support the case where GB_ENVIRONMENT env var is set but LAKEHOUSE_ENVIRONMENT env var is not
                lh = create_lakehouse_iceberg(
                    config="map",
                    conf_map=ConfigMap(environment=LAKEHOUSE_ENVIRONMENT, token=token),
                )
                __cache_lh("map", lh)
                logger.info(
                    "Loaded Lakehouse configuration from LAKEHOUSE_TOKEN env var for %s environment",
                    LAKEHOUSE_ENVIRONMENT,
                )
                log_lh_token_details(lh)
                return lh
            except Exception as e:
                error_str = (
                    f"failed to create a Lakehouse client from the config=map: {e}"
                )
        if lh is None:
            try:
                lh = create_lakehouse_iceberg(
                    config="env"
                )  # expects LAKEHOUSE_TOKEN/ENVIRONMENT env vars
                __cache_lh("env", lh)
                logger.info("Loaded Lakehouse configuration from env")
                log_lh_token_details(lh)
                return lh
            except Exception as e:
                error_str += (
                    f"\nfailed to create a Lakehouse client from the config=env: {e}"
                )
    raise ValueError(error_str)


class BaseLakehouseStorage(BaseModel):
    lh: Any = None  # Set in the initializer
    # lh_config_yaml: str = None
    """ 
    optional path to a yaml file holding lakehouse configuration.  If not specified, both the
    environment and lh-conf.yaml will be searched, in that order, for configuration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.lh is None:
            # self.lh = get_default_lh(config_yaml=self.lh_config_yaml)
            self.lh = get_default_lh()

    def delete_table_in_namespace(self: Self, namespace: str, table_name: str) -> None:
        """Delete the table, ignore if it doesn't exist.

        Args:
            namespace (str): _description_
            table_name (str): _description_
        """
        table = self.get_table_in_namespace(namespace, table_name)
        if table is not None:
            table.delete()

    def get_table_in_namespace(
        self: Self, namespace: str, table_name: str
    ) -> Table | None:
        """Get the named table in the given namespace, if it exists.

        Args:
            self (Self): _description_
            namespace (str): _description_
            table_name (str): _description_

        Returns:
            Table | None: None if table does not exist.
        """
        try:
            table = Table(lh=self.lh, namespace=namespace, table_name=table_name)
        except Exception:
            table = None
        return table


class BaseLakehouseItemStorage(BaseItemStorage, BaseLakehouseStorage):
    """
    Provides CRUD capabilities over pydantic BaseStoredItem objects in underlying Lakehouse storage.
    A given instance of this class is intended to be used with only one class of BaseStoredItem
    (e.g., Space, Artifact, etc).
    """

    namespace: str = GRANITE_DOT_BUILD_ADMIN_NAMESPACE
    # table_name: str
    # item_class: Type[BaseStoredItem]
    # lh: Any = None  # Set in the initializer
    # lh_config_yaml: str = None
    # logger: Any = None  # set in the initializer
    """ 
    optional path to a yaml file holding lakehouse configuration.  If not specified, both the
    environment and lh-conf.yaml will be searched, in that order, for configuration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = LoggingUtility(logger, msg_prefix=f"{self.table_name}")

    def __get_identity_partitions(self) -> list[str]:
        """
        Get the column names for which you would like partitions created.
        This is likely a sub-set of the columns returned by get_column_values().
        Be aware that over-paritioning can cause fragementation and degraded performance.
        """
        return None

    def _create_or_adjust_schema_item_dict(self, item: dict[str, Any]):
        """
        Create the table to match the given item and schema as defined by the columns/values defined in the given dictionary.
        column types should be derived from the types of the item fields.
        """
        # TODO: this is a bit inefficient since __get_table() is called again after this in subsequent calls.
        self.__get_table(item)  # This method creates the table and adjust the schemas

    def _add_item_dicts(self, items: list[dict[str, Any]]):
        """Called from add() after item validation and schema alignment to add the given list of 1 or more items as dictionaries to the database.

        Args:
            items (list[dict[str,Any]]): a list of BaseStoredItem converted to dictionaries.  Each dictionary includes
            the UUID_COLUMN_NAME, JSON_COLUMN_NAME and any other keys/values as defined by the sub-classes' _get_column_values(item) method.
        """
        df = self.__to_dataframe(items)
        table = self.__get_table(items)
        # table = Table(lh=self.lh, namespace=self.namespace, table_name=self.table_name) # DBG
        # pd.set_option('display.max_columns', None)
        # print(f"df={df}")
        try:
            table.append_dataframe(
                df=df,
                retry_on_conflict=True,
                max_retries=LAKEHOUSE_MAX_RETRIES,
                base_delay=LAKEHOUSE_BASE_DELAY,
            )
        except Exception as e:
            # DEBUGGING intermittent Access Denied
            msg = f"append_dtaframe upsert exception: '{table.namespace}'.'{table.name}'-'{table._catalog.cos.key}'"
            logger.error(msg)
            raise ValueError(msg) from e

    def _get_by_where_row_dicts(
        self,
        where: Optional[Union[str, dict]] = None,
        paginate: Optional[QueryControl] = None,
    ) -> list[dict[str, Any]]:
        """Called from get_by_were()
        Search for items via column values.
        The column values that are stored, and are therefore queryable,
        are defined by the sub-class implementation of _get_column_values(item)

        Args:
            where: if None, then get all.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): a dicitonary of columns names mapped to column values that will be used to build the WHERE clause by
            ANDing all the column=value expressions.

        Returns:
            list[dict[str,Any]]: list of matching dictionary item representations,  if any found, otherwise and empty list.
            Ordering of this list is undefined. dictionaries should be the same as the dictionaries received by
            _add_item_dict().
        """
        if paginate is not None:
            raise NotImplementedError("Pagination was requested, but is not supported")
        table = self.__get_table(None)
        if table is None:  # table not created yet, so there are no items to retrieve
            return {}
        item_dict = self.__get_by_row_filter(table, where)
        return item_dict

    def __get_row_filter_expression(
        self, where: Optional[dict]
    ) -> Optional[BooleanExpression]:
        from datetime import datetime

        where_expr = None
        if where is not None:
            for key, value in where.items():
                if isinstance(value, bool):
                    if value:
                        expr = EqualTo(key, value)
                    else:
                        # Allow Null as a match with False to accomodate schema changes that add a boolean column
                        expr = Or(EqualTo(key, value), IsNull(key))
                elif isinstance(value, datetime):
                    expr = EqualTo(key, value.isoformat())
                else:
                    expr = EqualTo(key, value)
                if where_expr is None:
                    where_expr = expr
                else:
                    where_expr = And(where_expr, expr)

        return where_expr

    def __get_by_row_filter(
        self, table: Table, row_filter: Optional[Union[str, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Use row_filter to query remote table.
        Use this if possible as it should be more efficient as only matching data is pulled and partitions will be used.

        Args:
            table (Table): _description_
            row_filter(Optional[Union[str, BooleanExpression, dict[str, Any]]]):
                str: an expression (sql?) specifying the matching rows
                dict[str,Any]: an expression will be built using AND and = from all the values
                BooleanExpression: an iceberg expression

        Returns:
            dict[str,dict[str,Any]]: list of dictionaries representing items.
        """
        self.logger.info(f"Searching for items using row_filter={row_filter}")
        if isinstance(row_filter, dict):
            row_filter = self.__get_row_filter_expression(row_filter)
        if row_filter is None:
            df = table.to_pandas()
        else:
            assert isinstance(row_filter, str) or isinstance(
                row_filter, BooleanExpression
            )
            df = table.to_pandas(row_filter=row_filter)

        items = self.__dataframe_to_dicts(df)
        self.logger.info(f"Found {len(items)} items")
        return items

    def __dataframe_to_dicts(self, df: DataFrame) -> list[dict[str, Any]]:
        """Convert each json column value to a BaseStoredItem using __from_json() and then key the items by their UUID into a dictionary"""
        items = []
        for _, row in df.iterrows():
            items.append(row)
        return items

    def _delete_table(self):
        table = self.__get_table(None)
        if table is not None:  # table exists.
            table.delete()

    def _upsert_item_dict(self, item: dict[str, Any]):
        """Called by update() after any created/updated_at_time fields are adjusted to add the dictionary of values.

        Args:
            item (dict[str,Any]): dictionary extraction of BaseStoredItem to be updated/upserted.  dictionary is created
            with the same semantics as add().

        Raises:
            ValueError: if no item with the given uuid exists.
            ValueError: if a field name in fields does not exist on the item.

        Returns:
            Optional[BASE_ITEM_TYPE]: The updated item if successful, or None if should_update
            returned False.
        """
        table = self.__get_table(item)
        df = self.__to_dataframe(item)

        assert isinstance(table, Table)
        # This always succeeds, or throws an exception, including RetryLimitExceeded exception
        try:
            table.upsert_dataframe(
                data=df,
                retry_on_conflict=True,
                max_retries=LAKEHOUSE_MAX_RETRIES,
                base_delay=LAKEHOUSE_BASE_DELAY,
            )
        except Exception as e:
            # DEBUGGGING intermittent Access Denied issue
            msg = f"upsert exception: '{table.namespace}'.'{table.name}'-'{table._catalog.cos.key}'"
            logger.error(msg)
            raise ValueError(msg) from e

    def update_fields(
        self,
        uuid: str,
        fields: dict[str, Any],
        should_update: Optional[Callable[[Any], bool]] = None,
        update_updated_time: bool = True,
    ) -> Optional[Any]:
        """Update the given fields of the item stored under the given uuid.

        WARNING: This Lakehouse implementation is NOT atomic. There is a race condition between
        reading the item, checking conditions, and writing the update.

        Args:
            uuid (str): id of item
            fields (dict[str,Any]): dictionary of field names and values to be replaced in the referenced item.
            should_update (Optional[Callable[[BASE_ITEM_TYPE], bool]]): If provided, a function that takes the
                current stored item and returns True if the update should proceed. If the function returns False,
                the update is NOT performed and None is returned.
            update_updated_time (bool, optional): whether to update the updated_time field. Defaults to True.

        Raises:
            ValueError: if no item with the given uuid exists.
            ValueError: if a field name in fields does not exist on the item.

        Returns:
            Optional[BASE_ITEM_TYPE]: The updated item if successful, or None if should_update
            returned False.
        """
        from gbserver.storage.storage import UPDATED_TIME_FIELD_NAME
        from gbserver.utils.utils import get_utc_time

        self.logger.info(
            f"Begin updating fields {list(fields.keys())} for item with uuid {uuid}"
        )
        item = self.get_by_uuid(uuid)
        if item is None:
            raise ValueError(
                f"Item with uuid {uuid} not found in table {self.table_name}"
            )

        # Check should_update condition
        if should_update is not None and not should_update(item):
            return None

        # Apply the field updates
        for field_name, field_value in fields.items():
            if not hasattr(item, field_name):
                raise ValueError(
                    f"Items of type {self.item_class.__name__} do not have an attribute named {field_name}"
                )
            setattr(item, field_name, field_value)
        if update_updated_time and hasattr(item, UPDATED_TIME_FIELD_NAME):
            setattr(item, UPDATED_TIME_FIELD_NAME, get_utc_time())

        # Convert item to row dict and upsert using Lakehouse
        item_dict = self._convert_item_to_row_dict(item)
        table = self.__get_table(item_dict)
        df = self.__to_dataframe(item_dict)

        assert isinstance(table, Table)
        try:
            table.upsert_dataframe(
                data=df,
                retry_on_conflict=True,
                max_retries=LAKEHOUSE_MAX_RETRIES,
                base_delay=LAKEHOUSE_BASE_DELAY,
            )
        except Exception as e:
            msg = f"upsert exception: '{table.namespace}'.'{table.name}'-'{table._catalog.cos.key}'"
            logger.error(msg)
            raise ValueError(msg) from e

        self.logger.info(f"Done updating fields for item with uuid {uuid}")
        return item

    def _delete(self, uuids: list[str]):
        """Delete the given ids.  Must handle if table has not been created (and ignore the request).
        Args:
            uuids (list[str]): _description_
        """
        table = self.__get_table(None)
        if table is not None:
            entries = []
            for id in uuids:
                ident = {UUID_COLUMN_NAME: id}
                entries.append(ident)
            table.delete_entries(entries=entries)

    def __to_dataframe(
        self, items: Union[dict[str, Any], list[dict[str, Any]]]
    ) -> DataFrame:
        if not isinstance(items, list):
            items = [items]
        # rows = []
        # for item in items:
        #     row = self._get_column_values(item)
        #     if row is None:
        #         raise ValueError(
        #             "Sub-class implementation of get_column_values() did not return a dictionary of column values for an item"
        #         )
        #     row[UUID_COLUMN_NAME] = item.uuid
        #     json = self.__to_json(item)
        #     row[JSON_COLUMN_NAME] = json
        #     rows.append(row)
        df = pd.DataFrame.from_dict(items)
        return df

    def __match_schema(self, table: Table, item: dict[str, Any]) -> bool:
        """Add/remove columns to the table as necessary to match the schema returned by get_column_values() on the given item.
        Args:
            table (Table): _description_
            item (BaseStoredItem): _description_
        Returns:
            bool: true if the schema has changed, in which case a new Table should be created.
        """
        row = item  # self._get_column_values(item)
        assert (
            row is not None
        ), "Sub-class did not return a dictionary of column values for an item"
        item_columns = list(row.keys())
        # df = table.to_pandas(rows=1)
        # existing_columns = list(df.columns)
        existing_columns = self.__get_column_names(table)

        # See if we need to add any columns
        changed = False
        for name in item_columns:
            if not name in existing_columns:
                self.__add_column(table, name, row[name])
                changed = True

        # # See if we need to delete any columns
        # existing_columns.remove(UUID_COLUMN_NAME)
        # existing_columns.remove(JSON_COLUMN_NAME)
        # for name in existing_columns:
        #     if not name in item_columns:
        #         self._remove_column(table, name)

        return changed

    def __add_column(self, table: Table, column_name: str, value: Any) -> None:
        type = StringType
        if isinstance(value, bool):
            type = BooleanType()
        elif isinstance(value, str):
            type = StringType()
        elif isinstance(value, float):
            type = FloatType()
        elif isinstance(value, int):
            type = IntegerType()
        elif isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                type = TimestampType()
            else:
                type = TimestamptzType()
        else:
            raise ValueError(f"Unsupported type for value {value}")

        iceberg_table = table.iceberg_table
        with iceberg_table.update_schema() as update:
            self.logger.info(f"Adding column {column_name} of type {type}")
            update.add_column(column_name, type)
            # TODO: There is an apparent bug in LH/Iceberg in that, if the column is moved, then subsequent appends give the following
            # (found when adding is_active to artifacts)

    #             if table_col.field_id != df_col.field_id:
    # >           raise TableSchemaIncompatibilityDifferentFieldId(
    #                         table_col.name, str(table_col.field_id), str(df_col.field_id))
    # E           lakehouse.exceptions.TableSchemaIncompatibilityDifferentFieldId: Dataframe schema has different fields compared with table schema.
    # E                                The column 'is_archived' has the wrong field_id. It should be '10' but it's '8'
    # See slack discussion : https://ibm-research.slack.com/archives/C05QM9A8WRG/p1744208489535329?thread_ts=1743896681.224139&cid=C05QM9A8WRG
    # update.move_before(column_name,UUID_COLUMN_NAME)
    # table.reload_iceberg_table()    # This was recommended to solve the problem above, but does not.

    def get_column_names(self) -> list[str]:
        return self.__get_column_names()

    def __get_column_names(self, table: Table = None) -> list[str]:
        """Get the list of column names of an existing table.
        Used primarily for testing.
        """
        if table is None:
            table = self.__get_table(None)
        return table.iceberg_table.schema().column_names

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=64),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def __create_Table(self) -> Table:
        try:
            table = Table(
                lh=self.lh, namespace=self.namespace, table_name=self.table_name
            )
        except Exception as e:
            self.logger.warning(f"Got exception trying to create Table: {e}")
            raise e
        return table

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=64),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _does_table_exist(self) -> bool:
        return self.lh._catalog.table_exists(self.namespace + "." + self.table_name)

    def __get_table(
        self, items: Union[dict[str, Any], list[dict[str, Any]]]
    ) -> Optional[Table]:
        """
        get the Table object and create if necessary.
        if item is None, then
            1) if the table exists, return it, else
            2) if the table does not exist, the return None.
        if item is NOT None, then never return None, and
            1) if the table exists, validate the schema and return the table.
            2) If the table does NOT exist, then create the table based on the item and return it.

        Raises
            ValueError if the table exists and the non-None item does not match the schema.
        """
        if isinstance(items, list):
            item: dict[str, Any] = items[0]
        else:
            item: dict[str, Any] = items
        table_exists = self._does_table_exist()
        if not table_exists:
            if item is None:
                table = None
            else:
                table = self.__create_table(item)
        else:
            table = self.__create_Table()

        if item and table:
            self.__match_schema(table, item)

        return table

    def __get_partition_columns(self) -> list[PartitionColumn]:
        names = self.__get_identity_partitions()
        if names is None or len(names) == 0:
            return []
        partitions = []
        for name in names:
            partitions.append(PartitionColumn(name=name, type="identity"))
        return partitions

    def __create_table(self, item=dict[str, Any]) -> Table:
        """
        Creates the non-existent table for our instance to store items to.

        Args:
            item (BaseStoredItem): representative item which we will be storing into the resulting table

        Returns:
            Table: table with schema for item's exposed columens, plus uuid and json columns.
        """
        if item is None:
            raise ValueError("Could not create table without item to define schema")
        self.logger.info(
            f"Begin creating table {self.table_name} in namespace {self.namespace}"
        )
        row = item
        for key, value in row.items():
            if value is None:
                raise ValueError(
                    f"Sub-class implementation of get_column_values() returned a None value for key {key}.  It must return a typed value to create the table."
                )
            if isinstance(value, list) or isinstance(value, dict):
                raise ValueError(
                    f"Sub-class implementation of get_column_values() returned a non-primitive value ({value}) for key {key}.  It must return a primitive value to create the table."
                )
        partitions = self.__get_partition_columns()
        df = self.__to_dataframe(item)
        table_properties = {
            "lh.maintenance-required": True,
            "lh.expire-snapshots-older-than-days": 1,
            "lh.rewrite-min-data-files": 3,
        }
        table_details = TableDetails(
            namespace=self.namespace,
            name=self.table_name,
            identifier_fields=[
                UUID_COLUMN_NAME
            ],  # TODO: does this keep us from adding a row with the same uuid?
            partition_fields=partitions,
            is_public=True,
            properties=table_properties,
        )
        # logger.info(f"df={df}")
        pa_table = pa.Table.from_pandas(df)
        # table = Table.from_dataframe(lh=self.lh, df=df, table_details=table_details)
        try:
            table = Table.from_arrow_schema(
                lh=self.lh, schema=pa_table.schema, table_details=table_details
            )
        except Exception as e:
            logger.error(
                f"Could not create table {self.namespace}.{self.table_name}: {e}"
            )
            raise e
        self.logger.info(
            f"Done creating table {self.table_name} in namespace {self.namespace}"
        )
        return table


if __name__ == "__main__":
    from gbserver.storage.stored_space import StoredSpace

    storage1 = BaseLakehouseItemStorage(
        table_name="gb-testing1", item_class=StoredSpace
    )
    storage2 = BaseLakehouseItemStorage(
        table_name="gb-testing2", item_class=StoredSpace
    )
    logger.info(f"Storage: {storage1}")
    logger.info(f"Storage: {storage2}")
