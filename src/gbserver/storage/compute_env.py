


from gbserver.storage.storage import BaseStoredItem


class StoredComputeEnv(BaseStoredItem):
    name: str 

    
if __name__ == "__main__":
    obj = StoredComputeEnv(name="foo", ignored=1)
    print(obj)