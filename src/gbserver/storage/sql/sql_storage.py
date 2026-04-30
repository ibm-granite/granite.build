import copy
import hashlib
import re
from datetime import datetime
from typing import Any, Callable, Dict, Generic, Optional, Self, Type, Union

from pydantic import Field
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    Unicode,
    UnicodeText,
    asc,
    desc,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError, NoSuchTableError, SQLAlchemyError
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
)

from gbserver.storage.sql.cert_file import get_ssl_cert_file
from gbserver.storage.sql.engine_cache import get_singleton_engine_cache
from gbserver.storage.storage import (
    BASE_ITEM_TYPE,
    JSON_COLUMN_NAME,
    UPDATED_TIME_FIELD_NAME,
    UUID_COLUMN_NAME,
    BaseItemStorage,
    QueryControl,
    SortOrder,
)
from gbserver.types.constants import (
    GBSERVER_SQL_DBNAME,
    GBSERVER_SQL_HOST,
    GBSERVER_SQL_PASSWD,
    GBSERVER_SQL_PORT,
    GBSERVER_SQL_SCHEMA,
    GBSERVER_SQL_SCHEME,
    GBSERVER_SQL_USER,
)
from gbserver.utils.atomic import AtomicInteger
from gbserver.utils.utils import get_utc_time

# SQLAlchemy Base
Base = declarative_base()

_CLASS_NAME_INDEX = AtomicInteger()

# Regex pattern for valid SQL identifiers (alphanumeric + underscore, cannot start with digit)
_VALID_SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_sql_identifier(name: str, identifier_type: str = "identifier") -> str:
    """Validate that a string is a safe SQL identifier to prevent SQL injection.

    Args:
        name: The identifier to validate (table name, column name, index name, etc.)
        identifier_type: Description of the identifier type for error messages

    Returns:
        The validated identifier (unchanged if valid)

    Raises:
        ValueError: If the identifier contains invalid characters
    """
    if not name or not _VALID_SQL_IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"Invalid SQL {identifier_type}: '{name}'. "
            f"Must contain only alphanumeric characters and underscores, and cannot start with a digit."
        )
    return name


class BaseSQLItemStorage(BaseItemStorage, Generic[BASE_ITEM_TYPE]):
    """
    Provides CRUD capabilities over pydantic BASE_ITEM_TYPE objects in underlying SQL storage.
    A given instance of this class is intended to be used with only one class of BASE_ITEM_TYPE
    (e.g., Space, Artifact, etc).
    """

    _db_schema: Optional[str]
    _db_url: str
    _obfuscated_db_url: str
    _connect_args: Optional[dict]

    _engine: Engine = None
    _inspector: Any = None
    _session_maker: Any = None
    _sql_alchemy_model: Any = None
    _column_types: dict[str, Any] = None

    _db_addr_hash: str  # = Field(init=False)
    """Internal hash of database addressing components"""

    unique_columns: dict[Union[str, tuple[str, ...]], Optional[Any]] = {}
    """
    Defined by the sub-class, as needed to set uniqueness attributes on columns/rows. Generally this contains zero or more of the
    column names returned by self._get_column_values(item), also defined by the sub-class.

    A dictionary where keys are column names (str) or tuples of column names for multi-column uniqueness.
    Values are exception values (or None for no exception). When an exception value is specified, rows with
    that value are exempt from the uniqueness constraint (implemented via partial unique indexes).

    Examples:
        {'checksum': ''} - checksum must be unique, but empty strings can be duplicated
        {'name': None} - name must always be unique (no exception)
        {('uri', 'space_name'): None} - the combination of uri and space_name must be unique
    """

    indexed_columns: list[str] = []
    """ A list of columns names (returned in _get_column_values()) that should be indexed."""

    exact_liked_list_columns: dict[str, str] = {}
    """Enables exact matching of a list of strings against a named column during get_by_where(dict) calls.
    The key is the column name in the query, and the value is name of the attribute on the item (containing a list value) 
    against which the matching is done. Typically, the column name and attribute name are the same.
    This is probably mostly used on columns in which %like% queries are used via the like_columns attribute above, but
    for which an exact match is desired on the string list values.  This was initially provided for tags on items
    which are stored in the column as "tag1,tag2,tag3..." with string type.  Using a combination of like and secondary filtering we 
    get the quicker exact match feature w/o having to commit to Postgres and it support for array type columns.
    """

    autoincr_column: str = None
    """The name of the column used to autoincrement the primary key"""

    default_pagination_sort_by_column: str = None
    """The name of the column to sort by when pagination is specified in the where query, if none is specified."""

    # Some implementation pointers...
    # 1. Use self.logger.info() to log message.  This will include the table name which is important.
    # 2. unique_fields, as supported in the super-class,  may not be needed if the db can enforce uniqueness (leave unset).
    # 3. The super class methods all log Begin/Done messages, so you may not need to do that here.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Validate table name and schema to prevent SQL injection - do this once at init time
        _validate_sql_identifier(self.table_name, "table name")

        self._db_schema, self._db_url, self._obfuscated_db_url, self._connect_args = (
            self._get_connection_specs()
        )
        if self._db_schema:
            _validate_sql_identifier(self._db_schema, "schema name")
        if not self._connect_args:
            self._connect_args = {}
        self.logger.info(
            f"Using schema '{self._db_schema}' and database connection {self._obfuscated_db_url}"
        )

        # Create a hash that is unique down to the dbschema
        self._db_addr_hash = hashlib.sha256(f"{self._db_url}".encode()).hexdigest()

        # These are set later when we know the item to be stored.
        self._engine = None
        self._inspector = None
        self._session_maker = None
        self._sql_alchemy_model = None
        self._column_types = None

    def _get_autoincr_column_type(self) -> Any:
        """By default, the auto increment column type is BigInteger (originally for postgres), but sub-classes
        can override to define another type, as is required for sqlite (Integer).
        The returned value will be used as the 'type' parameter to the Column initializer.
        """
        return BigInteger

    def _get_connection_specs(
        self,
    ) -> tuple[Optional[str], str, str, Optional[dict[str, str]]]:
        """Determines and returns the db connection specifications for this sub-class implementation.
        The default implementation uses  GBSERVER_SQL_* variables to define the database connection.

        Returns:
            tuple[str, str,str,dict[str,str]]: A set of connection information as follows:
                database schema to use - None if not used
                database connection URL - including password if needed.
                obfuscated database connection URL - db url w/o password
                connection args - a dictionary of arguments used when creating the db engine.
        """
        sql_scheme = GBSERVER_SQL_SCHEME
        host = GBSERVER_SQL_HOST
        port = GBSERVER_SQL_PORT
        user = GBSERVER_SQL_USER
        password = GBSERVER_SQL_PASSWD
        dbname = GBSERVER_SQL_DBNAME
        db_schema = GBSERVER_SQL_SCHEMA
        sslrootcert_file = get_ssl_cert_file(self.logger)

        if password is None or password == "":
            self.logger.warning(f"SQL password is not set")

        # Allow for an unspecified sslrootcert, but warn about it
        if sslrootcert_file is None:
            self.logger.warning(
                "SQL cert file is not set. Connection will be attempted w/o an SSL certificate."
            )
        else:
            self.logger.info(f"SQL cert file set to {sslrootcert_file}")

        connect_args = {"sslrootcert": sslrootcert_file} if sslrootcert_file is not None else {}
        db_url = f"{sql_scheme}://{user}:{password}@{host}:{port}/{dbname}"
        obfuscated_db_url = f"{sql_scheme}://{user}:**********@{host}:{port}/{dbname}"
        if sslrootcert_file is not None:
            db_url += "?sslmode=verify-full"
            obfuscated_db_url += "?sslmode=verify-full"
        return db_schema, db_url, obfuscated_db_url, connect_args

    def __get_table_args(self):
        """Build table args. Uniqueness constraints are handled via indexes in __create_unique_indexes()."""
        return ()

    def __create_sqlalchemy_class_from_dict(
        self, item: dict[str, Any], base: Type[declarative_base]
    ) -> tuple[Type[declarative_base], dict[str, Type]]:
        """
        Dynamically creates a SQLAlchemy declarative class from a python dictionary.
        Args:
            item: The dictionary to be saved via SQLAlchemy.
            base: The SQLAlchemy declarative base class (e.g., `DeclarativeBase` or `Base = declarative_base()`).
            schemma: The schema name.
            tablename: table name should be lower case
        Returns:
            A new SQLAlchemy declarative instance representing the Pydantic model.
        """
        tablename = self.table_name.lower()
        column_types: Dict[str, Any] = {}
        table_args = self.__get_table_args()
        attributes: Dict[str, Any] = {"__tablename__": tablename}
        if self._db_schema:
            attributes["__table_args__"] = table_args + (
                {"schema": self._db_schema, "extend_existing": True},
            )
        else:
            attributes["__table_args__"] = table_args + ({"extend_existing": True},)
        # Make this the first column in the view
        column_types[UUID_COLUMN_NAME] = String(128)
        if self.autoincr_column is None:
            attributes[UUID_COLUMN_NAME] = Column(String(128), primary_key=True)
        else:
            attributes[UUID_COLUMN_NAME] = Column(
                String(128), primary_key=False, index=True, unique=True
            )
            # attributes[self.autoincr_column] = Column(BigInteger, Identity(start=0, cycle=True), primary_key=True)
            col_type = self._get_autoincr_column_type()
            attributes[self.autoincr_column] = Column(
                col_type, primary_key=True, autoincrement=True
            )
            column_types[self.autoincr_column] = col_type

        # Iterate through each field defined in dictionary
        hash_source = ""
        for key, value in item.items():
            if key == UUID_COLUMN_NAME:
                continue
            if key == JSON_COLUMN_NAME:
                continue

            unique = key in self.unique_columns
            indexed = key in self.indexed_columns

            # Get the Python type of the value.
            lower_key: str = key.lower()

            if (
                lower_key == "uri"
                or lower_key == "url"
                or "uri_" in lower_key
                or "_uri" in lower_key
            ):
                # This one needs to be longer than 256, sometimes.
                column = Column(String(1024), nullable=True, index=indexed)
            elif isinstance(value, str):
                column = Column(String(256), nullable=True, index=indexed)
            elif isinstance(value, bool):
                column = Column(Boolean, nullable=True, index=indexed)
            elif isinstance(value, int):
                column = Column(Integer, nullable=True, index=indexed)
            elif isinstance(value, float):
                column = Column(Float, nullable=True, index=indexed)
            elif isinstance(value, datetime):
                column = Column(DateTime(timezone=True), nullable=True, index=indexed)
            else:
                column = Column(String(256), nullable=True, unique=unique, index=indexed)
            attributes[key] = column
            column_types[key] = column.type
            hash_source = hash_source + key

        # Try and make this the last column in the views
        column_types[JSON_COLUMN_NAME] = Text
        attributes[JSON_COLUMN_NAME] = Column(Text)

        # Make sure we never have a type name collision, especially for the same tables
        # that are accessed by more than one instance of this class.
        type_name_disambigutator = str(_CLASS_NAME_INDEX.fetch_and_add())
        type_name = "sql_orm_" + str(type_name_disambigutator)
        # self.logger.info(f"sqlalchemy dynamic type created with name {type_name} and attributes {attributes}")
        sqlalchemy_model = type(type_name, (base,), attributes)
        return sqlalchemy_model, column_types

    def __initialize_model_and_table(self, item_dict: dict[str, Any], re_init: bool = False):
        if self._sql_alchemy_model is None or re_init:
            self._sql_alchemy_model, self._column_types = self.__create_sqlalchemy_class_from_dict(
                item_dict, Base
            )

        # Always make sure the table exists since delete_table() could be followed by add() (mostly from tests though).
        if not self._does_table_exist() or re_init:
            try:
                self.__connect_with_retry()  # To create engine and inspector
                self._inspector.clear_cache()
                self._sql_alchemy_model.__table__.create(
                    self._engine
                )  # Create only this one table among N known by Base.metadata
                self.logger.info(
                    f"Table '{self._sql_alchemy_model.__tablename__}' created successfully."
                )
                self.__create_unique_indexes()
            except SQLAlchemyError as e:
                self.logger.error(f"Error creating table: {e}")
                raise e

    def __create_unique_indexes(self):
        """Create unique indexes for columns specified in unique_columns."""
        if not self.unique_columns:
            return

        try:
            with self._engine.connect() as connection:
                for column_key, exception_value in self.unique_columns.items():
                    statement = self.__get_unique_index_statement(column_key, exception_value)
                    connection.execute(text(statement))
                connection.commit()
        except Exception as e:
            self.logger.warning(f"Error creating unique indexes: {e}")

    def __get_unique_index_statement(
        self, column_key: Union[str, tuple[str, ...]], exception_value: Optional[Any]
    ) -> str:
        """Generate the SQL statement to create a unique index.

        Args:
            column_key: Column name (str) or tuple of column names for multi-column uniqueness.
            exception_value: If None, creates a standard unique index. If not None, creates a partial
                unique index that only enforces uniqueness for values not equal to the exception.

        Returns:
            SQL statement to create the unique index.
        """
        # Determine column names for the index and validate them
        if isinstance(column_key, tuple):
            for col in column_key:
                _validate_sql_identifier(col, "column name")
            col_suffix = "_".join(column_key)
            column_list = ", ".join(column_key)
        else:
            _validate_sql_identifier(column_key, "column name")
            col_suffix = column_key
            column_list = column_key

        # PostgreSQL limits identifiers to 63 bytes.  Build a name that is always
        # unique and always fits:  "uq_" + 7-char SHA1 hash of table+columns + "_" + col_suffix,
        # truncated to 63 chars.
        full_name = f"uq_{self.table_name}_{col_suffix}"
        if len(full_name) > 63:
            name_hash = hashlib.sha1(full_name.encode()).hexdigest()[:7]
            index_name = f"uq_{name_hash}_{col_suffix}"[:63]
        else:
            index_name = full_name

        if exception_value is None:
            # Standard unique index
            return f"CREATE UNIQUE INDEX {index_name} ON {self.__get_sql_table_name_reference()} ({column_list});"
        else:
            # Partial unique index - only enforce uniqueness when value != exception
            # Note: Only single-column partial indexes are supported
            if isinstance(column_key, tuple):
                self.logger.warning(
                    f"Exception values not supported for multi-column uniqueness: {column_key}"
                )
                return f"CREATE UNIQUE INDEX {index_name} ON {self.__get_sql_table_name_reference()} ({column_list});"
            else:
                # Escape single quotes to prevent SQL injection (double them for SQL)
                escaped_value = str(exception_value).replace("'", "''")
                return f"CREATE UNIQUE INDEX {index_name} ON {self.__get_sql_table_name_reference()} ({column_list}) WHERE {column_key} != '{escaped_value}';"

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def __connect_with_retry(self):
        self.__connect_without_retry()

    def __connect_without_retry(self):
        if self._session_maker is None:
            # if random.randint(0,4) > 0:
            #     raise ValueError(f"Simulating connection failure on {self.obfuscated_db_url}")
            try:
                self._engine = get_singleton_engine_cache().get_engine(
                    db_uri=self._db_url,
                    connect_args=self._connect_args,
                    pool_use_lifo=True,
                    pool_size=20,
                    max_overflow=0,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
                self._inspector = inspect(self._engine)
                self._session_maker = sessionmaker(
                    autocommit=False, autoflush=False, bind=self._engine
                )
                self._scoped_session = scoped_session(self._session_maker)
            except Exception as e:
                raise ValueError(
                    f"Could not create engine/inspector/sessionmaker for {self._obfuscated_db_url}: {e}"
                )

    def __get_session_without_retry(self) -> Any:
        self.__connect_without_retry()
        # return self._session_maker()
        return self._scoped_session()

    def _create_or_adjust_schema_item_dict(self, item: dict[str, Any], re_init: bool = False):
        """
        Create the table to match the given item and schema as defined by the columns/values defined in the given dictionary.
        column types should be derived from the types of the item fields.
        """

        try:
            # NOTE: if the table already exists, it is NOT modified to match the columns indicated by the item.
            # Thus the need to check the table:item match and adjust as necessary, below.
            self.__initialize_model_and_table(item, re_init)
        except SQLAlchemyError as e:
            self.logger.error(f"Error creating table: {e}")
            raise e

        new_columns = []
        existing_columns = self.__get_column_names_with_exceptions(raise_exception=False)
        self.logger.info(f"Columns found in the table: {existing_columns}")
        for column_name in list(item.keys()):
            if column_name not in existing_columns:
                new_columns.append(column_name)
                self.logger.info(f"New column to be added: {column_name}")

        if len(new_columns) > 0 and len(existing_columns) > 0:
            item_copy = copy.deepcopy(item)
            for new_col in new_columns:
                del item_copy[new_col]
            for col_name in new_columns:
                col_type = self._column_types[col_name]
                is_indexed = col_name in self.indexed_columns
                uniqueness_exception = (
                    self.unique_columns.get(col_name) if col_name in self.unique_columns else None
                )
                is_unique = col_name in self.unique_columns
                self.__add_column(col_name, col_type, is_indexed, is_unique, uniqueness_exception)

    def __get_sql_table_name_reference(self) -> str:
        """Get the name of the schema, if any, concatenated with the table name for using in SQL statements.
        Note: table_name and schema are validated at __init__ time to prevent SQL injection.
        """
        if self._db_schema:
            table_ref = self._db_schema + "." + self.table_name
        else:
            table_ref = self.table_name
        return table_ref

    def __add_column(
        self,
        col_name: str,
        col_type: type,
        is_indexed: bool,
        is_unique: bool,
        uniqueness_exception: Optional[Any],
    ):
        # Validate column name to prevent SQL injection
        _validate_sql_identifier(col_name, "column name")

        column = Column(name=col_name, type_=col_type, nullable=True)
        column_name = column.compile(dialect=self._engine.dialect)
        column_type = column.type.compile(self._engine.dialect)
        self._inspector.clear_cache()
        try:
            with self._engine.connect() as connection:
                statement = f"ALTER TABLE {self.__get_sql_table_name_reference()} ADD {column_name} {column_type};"
                connection.execute(text(statement))
                if is_indexed:
                    index_name = f"ix_{self.table_name}_{col_name}"
                    index_statement = f"CREATE INDEX {index_name} ON {self.__get_sql_table_name_reference()} ({column_name});"
                    connection.execute(text(index_statement))
                if is_unique:
                    unique_statement = self.__get_unique_index_statement(
                        col_name, uniqueness_exception
                    )
                    connection.execute(text(unique_statement))
                connection.commit()
        except Exception as e:
            self.logger.warning(f"Error adding column: {e}")

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        retry=retry_if_not_exception_type(IntegrityError),
        reraise=True,
    )
    def _add_item_dicts(self, items: list[dict[str, Any]]):
        """Called from add() after item validation and schema alignment to add the given list of 1 or more items as dictionaries to the database.

        Args:
            items (list[dict[str,Any]]): a list of BASE_ITEM_TYPE converted to dictionaries.  Each dictionary includes
            the UUID_COLUMN_NAME, JSON_COLUMN_NAME and any other keys/values as defined by the sub-classes' _get_column_values(item) method.

        """
        session = self.__get_session_without_retry()
        try:
            if len(items) > 5:
                batch_size = 100
                batch_insert = []
                for item_data in items:
                    db_item = self._sql_alchemy_model(**item_data)
                    batch_insert.append(db_item)
                    if len(batch_insert) >= batch_size:
                        session.bulk_save_objects(batch_insert)
                        batch_insert = []
                if len(batch_insert) > 0:
                    session.bulk_save_objects(batch_insert)
            else:
                for item_data in items:
                    self.logger.info(f"_add: {item_data}")
                    db_item = self._sql_alchemy_model(**item_data)
                    session.add(db_item)
            session.commit()
            self.logger.info(f"{len(items)} objects added to table")
        except IntegrityError as e:
            session.rollback()
            self.logger.error(f"Error inserting item into table (IntegrityError): {e}")
            raise e
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error inserting item to table: {e}")
            raise e
        finally:
            session.close()

    # Don't need @retry here since we have it on get_by_where() above.
    def _get_by_where_row_dicts(
        self,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[dict[str, Any]]:
        """Called from get_by_where()
        Search for items via column values.
        The column values that are stored, and are therefore queryable,
        are defined by the sub-class implementation of _get_column_values(item)

        Args:
            where: if None, then get all.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): a dicitonary of columns names mapped to column values that will be used to build the WHERE clause by
            ANDing all the column=value expressions.
            query_control[dict[str,int]]: If provide specifies keys 'index' and 'size' controlling the zer-based page index and the number of rows in a page.
            if specified, the ordering of the list should be such that the most recently added rows are in the first page.

        Returns:
            list[dict[str,Any]]: list of matching dictionary item representations,  if any found, otherwise and empty list.
            Ordering of this list is undefined. dictionaries should be the same as the dictionaries received by
            _add_item_dict().
        """

        session = self.__get_session_without_retry()
        try:
            r = self.__get_by_where_row_dicts_with_session(session, where, query_control)
            return r
        finally:
            session.close()

    def __get_by_where_row_dicts_with_session(
        self,
        session: Any,
        where: Optional[Union[str, dict]] = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[dict[str, Any]]:
        # return selected rows from the where dictionary
        try:
            results = []
            query = self.__get_where_query(session, where=where, query_control=query_control)
            query_results = query.all()
            for query_result in query_results:
                package_dict = {}
                for k, v in self._column_types.items():
                    # Cast back to original data type, recorded in self.attribute_type
                    # package_dict[k] = v(getattr(query_result, k))
                    package_dict[k] = getattr(query_result, k)
                results.append(package_dict)
            return results
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error querying table: {e}")
            raise e

    def __split_where_query(self, where) -> tuple[dict, dict]:
        exact = {}
        likes = {}
        for column, value in where.items():
            if column in self.exact_liked_list_columns:
                likes[column] = value
            else:
                exact[column] = value
        return exact, likes

    def __get_where_query(
        self: Self,
        session,
        where: Optional[Union[dict[str, Any], str]],
        query_control=Optional[QueryControl],
    ):
        if where is None:
            query = session.query(self._sql_alchemy_model)
        elif isinstance(where, dict):
            and_where, like_where = self.__split_where_query(where)
            # Create a filter checking the type of the colum and the type of the given value:
            # For each key/value:
            #    If the model column is a string-like column and the value is a list/array, use IN
            #    Otherwise, use equality (==)
            filters = []
            for key, value in and_where.items():
                column = getattr(self._sql_alchemy_model, key)

                # Check if column is string-like
                is_string_column = isinstance(column.type, (String, Unicode, UnicodeText))

                # If value is a list/tuple and column is string → use IN
                if is_string_column and isinstance(value, (list, tuple, set)):
                    filters.append(column.in_(value))
                else:
                    filters.append(column == value)

            query = session.query(self._sql_alchemy_model).filter(*filters)
            for col, like_value in like_where.items():
                if isinstance(like_value, str):
                    like_value = [like_value]
                elif isinstance(like_value, list):
                    pass
                else:
                    raise Exception('Invalid type "like_value".  Must be one of str or list[str].')
                for value in like_value:
                    query = query.filter(getattr(self._sql_alchemy_model, col).like(f"%{value}%"))
        else:
            assert isinstance(where, str)
            raise NotImplementedError("WHERE claused based queries not supported (yet).")
        if query_control is not None:
            query = self.__control_query(query, query_control)
        return query

    def __control_query(self, query, query_control: QueryControl):
        sort_orders = query_control.sort_orders
        if not sort_orders:
            sort_orders = []
            if self.default_pagination_sort_by_column:
                sort_orders.append(
                    SortOrder(column=self.default_pagination_sort_by_column, ascending=True)
                )
        for sort_order in sort_orders:
            order_by = sort_order.column
            if sort_order.ascending:
                query = query.order_by(asc(order_by))
            else:
                query = query.order_by(desc(order_by))

        pagination = query_control.pagination
        if pagination:
            assert pagination.index >= 0
            assert pagination.size > 0
            offset = pagination.index * pagination.size
            query = query.limit(pagination.size).offset(offset)
        return query

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _delete_table(self):
        """Delete the table in self.table_name. Ignore when table does not exist"""

        if self._sql_alchemy_model is None:
            if self._does_table_exist():
                # Table exists, but this instance did not created, but wants to delete it
                try:
                    self.__connect_without_retry()
                    with self._engine.connect() as connection:
                        connection.execute(
                            text(f"DROP TABLE {self.__get_sql_table_name_reference()}")
                        )
                        connection.commit()  # Commit the transaction to apply changes
                    self.logger.info(
                        f"Table {self.__get_sql_table_name_reference()} deleted successfully."
                    )
                except SQLAlchemyError as e:
                    self.logger.error(f"Error deleting table: {e}")  # When table does not exist.
                    raise e
        try:
            self.__connect_without_retry()  # Ensure engine is initialized
            self._sql_alchemy_model.__table__.drop(self._engine)
            self.logger.info(f"Table {self.__get_sql_table_name_reference()} deleted successfully.")
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting table: {e}")  # When table does not exist.
            raise e

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _does_table_exist(self) -> bool:
        columns = self.__get_column_names_with_exceptions(
            raise_exception=False
        )  # returns [] if tables is not present
        return len(columns) > 0

    def __get_db_item_by_uuid(self, session: Any, uuid: str) -> Optional[BASE_ITEM_TYPE]:
        """Get the SQLAlchemy database item corresponding to the item with the given UUID.
        Since UUID is not always the primary key, we use filter_by() instead of get().

        Args:
            session (Any): _description_
            item_uuid (str): _description_

        Returns:
            Optional[BASE_ITEM_TYPE]: _description_
        """
        # item_to_delete = session.query(self.sql_alchemy_model).get(uuid)
        where = {}
        where[UUID_COLUMN_NAME] = uuid
        db_item = session.query(self._sql_alchemy_model).filter_by(**where).first()
        return db_item

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _delete(self, uuids: list[str]):
        """Delete the given ids, ignore if table does not exist.
        Args:
            uuids (list[str]): _description_
        """
        session = self.__get_session_without_retry()
        try:
            deleted_uuids = []
            for uuid in uuids:
                item_to_delete = self.__get_db_item_by_uuid(session=session, uuid=uuid)
                if item_to_delete:
                    session.delete(item_to_delete)
                    deleted_uuids.append(uuid)
                else:
                    self.logger.warning(f"Row with UUID {uuid} not found.")
            # Commit all deletes in a single transaction
            if deleted_uuids:
                session.commit()
                self.logger.info(f"Deleted {len(deleted_uuids)} row(s) with UUIDs: {deleted_uuids}")
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error deleting rows: {e}")
            raise e
        finally:
            session.close()

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _get_column_names(self) -> list[str]:
        """Implemented to return the columns of an existing table, so we don't expect exceptions"""
        return self.__get_column_names_with_exceptions(raise_exception=True)

    def __get_column_names_with_exceptions(self, raise_exception: bool) -> list[str]:
        """Get the list of columns or an empty list if the table does not exist and raise_exception=False.

        Returns:
            list[str]: _description_
        """
        existing_columns = []
        try:
            self.__connect_with_retry()  # To create inspector
            self._inspector.clear_cache()
            for column in self._inspector.get_columns(self.table_name, schema=self._db_schema):
                existing_columns.append(column["name"])
        except NoSuchTableError as e:
            if raise_exception:
                raise e
        return existing_columns

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def get_by_where(
        self,
        where: str | dict | None = None,
        query_control: Optional[QueryControl] = None,
    ) -> list[BASE_ITEM_TYPE]:
        """Override the super-class method to add support for like-style queries on the exact_liked_columns."""
        items = super().get_by_where(where, query_control=query_control)
        if isinstance(where, dict):
            # For queries, such as %like%, we can enable better exact list member match here via exact_liked_list_columns.
            # Yes, this would be better to do directly in SQL, but we didn't want to commit to SQL or Mongo to enable this feature.
            # We sacrifices performance for flexibility in SQL provider.
            for column_name, attr_name in self.exact_liked_list_columns.items():
                list_values_to_match = where.get(column_name, None)  # column of interest is queried
                if list_values_to_match is not None:
                    # Get the items from the where query that are all found in the list value under the named attribute on the item.
                    items = self.__filter_by_list_values(items, attr_name, list_values_to_match)
        return items

    def __filter_by_list_values(
        self,
        items: list[BASE_ITEM_TYPE],
        list_attr_name: str,
        list_values_to_match: Union[list[str], str],
    ) -> list[BASE_ITEM_TYPE]:
        """Filter the items to find those that have an atttribute with a list value that matches the given list of values to match.
        The attribute list value must contain all of the values in the given list, but may contain more.
        This is initially provided in support of the tags column for artifacts that stores the tags in a column as a string of comma-separated tags.
        We search that column with %like% queries which may get more than the exact matches.  This can/should be used to further filter to get exact matches.

        Args:
            items (list[BASE_ITEM_TYPE]): items to filter.
            list_attr_name (str): name of the attribute on the items contain a list value to search for matching values.
            list_values_to_match (Union[list[str],str]): A list of value all of which must appear in the attribute value list to be considered a match.
            None, [], or '' may be be used here to specify that the matching items should have no tags (i.e. attribute list value is None or an empty list).

        Returns:
            list[BASE_ITEM_TYPE]: A list of zero or more matching items.
        """
        if isinstance(list_values_to_match, str):
            list_values_to_match = [list_values_to_match]
        if len(list_values_to_match) > 0:
            matched_items = []
            for item in items:
                matched = True
                for tag in list_values_to_match:
                    item_attr_list_value = getattr(item, list_attr_name)
                    if (len(tag) == 0 or tag is None) and (
                        item_attr_list_value is None or len(item_attr_list_value) == 0
                    ):  # Searching for an empty tag ('') matches an empty list of tags.
                        continue
                    assert isinstance(item_attr_list_value, list)
                    if not tag in item_attr_list_value:
                        matched = False
                        break
                if matched:
                    matched_items.append(item)
            items = matched_items
        return items

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        reraise=True,
    )
    def _count(self, where: Optional[Union[str, dict]] = None) -> int:
        """Return the number of rows in the table matching the where clause.

        Args:
            where: if None, count all items.
            where(str): SQL WHERE clause (w/o the WHERE).
            where(dict): dictionary of column names mapped to values for WHERE clause.

        Returns:
            int: the count of matching rows.
        """
        session = self.__get_session_without_retry()
        try:
            query = self.__get_where_query(session, where=where, query_control=None)
            count = query.count()
            return count
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error counting rows in table: {e}")
            raise e
        finally:
            session.close()

    @retry(
        wait=wait_random_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(10),
        retry=retry_if_not_exception_type((ValueError, IntegrityError)),
        reraise=True,
    )
    def update_fields(
        self,
        uuid: str,
        fields: dict[str, Any],
        should_update: Optional[Callable[[BASE_ITEM_TYPE], bool]] = None,
        update_updated_time: bool = True,
    ) -> Optional[BASE_ITEM_TYPE]:
        """Update the given fields of the item stored under the given item uuid.
        The implementation uses SELECT FOR UPDATE to ensure true atomicity - the row is locked
        during read, preventing other transactions from modifying it until this transaction completes.

        Args:
            uuid (str): id of item
            fields (dict[str,Any]): dictionary of field names and values to be replaced in the referenced item.
            should_update (Optional[Callable[[BASE_ITEM_TYPE], bool]]): If provided, a function that takes the
                current stored item and returns True if the update should proceed. The check and update are
                performed atomically within the same transaction with row-level locking. If the function
                returns False, the update is NOT performed and None is returned.
            update_updated_time (bool, optional): whether to update the updated_time field. Defaults to True.

        Raises:
            ValueError: if fields are not present on the items stored in this instance.
            ValueError: if the uuid given is not found.
            ValueError: if should_update is provided and it throws an exception.

        Returns:
            Optional[BASE_ITEM_TYPE]: The updated item if the update was performed, or None if
            should_update was provided and returned False.
        """
        session = self.__get_session_without_retry()
        try:
            # Use SELECT FOR UPDATE to lock the row during read - prevents race conditions
            stmt = (
                select(self._sql_alchemy_model)
                .where(getattr(self._sql_alchemy_model, UUID_COLUMN_NAME) == uuid)
                .with_for_update()
            )

            result = session.execute(stmt)
            db_item = result.scalar_one_or_none()

            if db_item is None:
                raise ValueError(f"Item with id {uuid} not found.")

            # Convert db_item to our item type for should_update check
            # Use self._column_types to match how __get_by_where_row_dicts_with_session extracts data
            item_dict = {k: getattr(db_item, k) for k in self._column_types.keys()}
            item = self._convert_row_dict_to_item(item_dict)

            # Check should_update condition (row is locked, so this is truly atomic)
            if should_update is not None:
                try:
                    if not should_update(item):
                        # Row lock is released when session closes
                        return None
                except Exception as se:
                    raise ValueError("Exception during item test.") from se

            # Validate and apply the field updates to the Pydantic item
            for field_name, field_value in fields.items():
                if not hasattr(item, field_name):
                    raise ValueError(f"Field {field_name} not found in item")
                if field_name in [UUID_COLUMN_NAME, JSON_COLUMN_NAME]:
                    raise ValueError(f"Field {field_name} can not be updated")
                setattr(item, field_name, field_value)

            if update_updated_time and hasattr(item, UPDATED_TIME_FIELD_NAME):
                setattr(item, UPDATED_TIME_FIELD_NAME, get_utc_time())

            # Convert the updated item to a row dict and use it to update all db_item fields
            # This ensures JSON column and all derived columns are properly updated
            updated_item_dict = self._convert_item_to_row_dict(item)
            for column_name, column_value in updated_item_dict.items():
                setattr(db_item, column_name, column_value)

            # Commit the transaction and apply the changes to db_item - this releases the row lock
            session.commit()
            return item
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Error in update_fields: {e}")
            raise e
        finally:
            session.close()
