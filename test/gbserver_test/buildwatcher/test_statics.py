from gbcommon.uri.lh import DEFAULT_FILESET_VERSION, DEFAULT_MODEL_REVISION
from gbserver.buildwatcher.buildrunner import BuildRunner


class TestBuildWatcherStatics:

    def test_get_normalized_uri(self):
        namespace = "mynamespace"
        table_name = "mytable"
        name = "myfileset"
        rev = "1"

        uri = f"lh://anything/{namespace}/filesets/{table_name}/{name}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(DEFAULT_FILESET_VERSION)
        uri = f"lh://anything/{namespace}/filesets/{table_name}/{name}/{rev}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(rev)

        uri = f"lh://anything/{namespace}/models/{table_name}/{name}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(DEFAULT_MODEL_REVISION)
        uri = f"lh://anything/{namespace}/models/{table_name}/{name}/{rev}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(rev)
