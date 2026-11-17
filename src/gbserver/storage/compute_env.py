# SPDX-License-Identifier: Apache-2.0

from gbserver.storage.storage import BaseStoredItem


class StoredComputeEnv(BaseStoredItem):
    name: str


if __name__ == "__main__":
    obj = StoredComputeEnv(name="foo")
    print(obj)
