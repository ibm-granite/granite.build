# Recommended to run this with "export GBSERVER_KEEP_UPDATED_TIME=TRUE"
import sys

from gbserver.storage.singleton_storage import get_admin_storage

storage = get_admin_storage()

builds = storage.build_storage.get_by_where({})

for b in builds:
    changed = b.compact_build_archive()
    if changed:
        print(f"updated build_archive for {b.uuid}")
        storage.build_storage.update(item=b)
