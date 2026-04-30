"""Compute env module."""

from gbserver.storage.storage import BaseStoredItem


class StoredComputeEnv(BaseStoredItem):
    """Stored Compute Env implementation."""

    name: str


if __name__ == "__main__":
    obj = StoredComputeEnv(name="foo")
    print(obj)
