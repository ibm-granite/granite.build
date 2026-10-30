from typing import Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def _import_lakehouse_iceberg():
    """Lazy import LakehouseIceberg. Raises ImportError with clear message if not installed."""
    try:
        from lakehouse import LakehouseIceberg

        return LakehouseIceberg
    except ImportError as e:
        raise ImportError(
            "The 'lakehouse' (dmf-lib) library is required for Lakehouse operations. "
            "Install it with: pip install dmf-lib"
        ) from e


@retry(
    wait=wait_exponential(multiplier=2, min=2, max=64),
    stop=stop_after_attempt(5),
    # A missing lakehouse library is permanent; don't spend ~30s backing off.
    # Keep the default Exception filter so KeyboardInterrupt isn't retried.
    retry=retry_if_exception_type(Exception) & retry_if_not_exception_type(ImportError),
    reraise=True,
)
def create_lakehouse_iceberg(
    config: str = "yaml",
    conf_location: Optional[str] = None,
    conf_map: Optional[dict] = None,
):
    """Create a LakehouseIceberg instance with retry logic.

    This function wraps LakehouseIceberg instantiation with exponential backoff
    retries to handle transient connection failures.

    Args:
        config: Configuration type - "yaml", "map", or "env". Defaults to "yaml".
        conf_location: Path to a configuration file.
        conf_map: ConfigMap instance when config="map".

    Returns:
        LakehouseIceberg: A configured Lakehouse client instance.

    Raises:
        Exception: Re-raises the last exception after all retries are exhausted.
    """
    LakehouseIceberg = _import_lakehouse_iceberg()
    return LakehouseIceberg(
        config=config,
        conf_location=conf_location,
        conf_map=conf_map,
    )
