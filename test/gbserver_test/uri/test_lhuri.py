from urllib.parse import urlparse

import pytest

from gbcommon.uri.lh import (
    DEFAULT_FILESET_VERSION,
    DEFAULT_MODEL_REVISION,
    LhType,
    LhURI,
)
from gbcommon.uri.uri import URI


def test_model_uri():
    table_name = "mytable"
    name = "mymodel"
    revision = "123"
    uri = LhURI.get_model_uri(table_name=table_name, model_label=name, model_revision=revision)
    assert uri.endswith(f"models/{table_name}/{name}/{revision}")
    assert table_name in uri, "Table name did not get included"
    assert name in uri, "Name did not get included"
    assert revision in uri, "Revision did not get included"
    parse = urlparse(uri)
    lh = LhURI(parse)
    assert lh.get_lh_table_name() == table_name
    assert lh.get_lh_model_label() == name
    assert lh.get_lh_model_revision() == revision
    md = lh.get_metadata()
    assert md["type"] == "model"
    assert md["table_name"] == table_name
    assert md["model_label"] == name
    assert md["model_revision"] == revision
    assert md.get("dataset_name") == None


def test_model_uri_no_revision():
    namespace = "mynamespace"
    table_name = "mytable"
    name = "mymodel"
    revision = ""
    uri = LhURI.get_model_uri(table_name=table_name, model_label=name)
    assert uri.endswith(f"models/{table_name}/{name}/{DEFAULT_MODEL_REVISION}")
    assert table_name in uri, "Table name did not get included"
    assert name in uri, "Name did not get included"
    uri = f"lh://somehost/{namespace}/models/{table_name}/{name}"
    parse = urlparse(uri)
    lh = LhURI(parse)
    assert lh.get_lh_table_name() == table_name
    assert lh.get_lh_model_label() == name
    assert lh.get_lh_model_revision() == DEFAULT_MODEL_REVISION
    md = lh.get_metadata()
    assert md["type"] == "model"
    assert md["table_name"] == table_name
    assert md["model_label"] == name
    assert md["model_revision"] == DEFAULT_MODEL_REVISION
    assert md.get("dataset_name") == None


def test_model_uri_no_table():
    def test_helper():
        namespace = "mynamespace"
        table_name = ""
        name = "mymodel"
        revision = "123"
        uri = LhURI.get_model_uri(table_name=table_name, model_label=name)
        assert uri.endswith(f"models/{table_name}/{name}/{DEFAULT_MODEL_REVISION}")
        assert table_name in uri, "Table name did not get included"
        assert name in uri, "Name did not get included"
        uri = f"lh://somehost/{namespace}/models/{table_name}/{name}"
        parse = urlparse(uri)
        lh = LhURI(parse)

    with pytest.raises(Exception) as e:
        test_helper()
    assert (
        "failed to create from uri: ParseResult(scheme='lh', netloc='somehost', path='/mynamespace/models//mymodel', params='', query='', fragment='')"
        in str(e.value)
    )
    nested_e = e.value.__cause__
    assert isinstance(nested_e, ValueError)
    assert "The table name cannot be empty in a lh:// URI" in str(nested_e)


def test_dataset_uri():
    table_name = "mytable"
    dataset_name = "mymodel"
    uri = LhURI.get_dataset_uri(table_name=table_name, dataset_name=dataset_name)
    assert uri.endswith(f"datasets/{table_name}/{dataset_name}")
    assert table_name in uri, "Table name did not get included"
    assert dataset_name in uri, "Name did not get included"
    parse = urlparse(uri)
    lh = LhURI(parse)
    assert lh.get_lh_table_name() == table_name
    assert lh.get_lh_dataset_name() == dataset_name
    md = lh.get_metadata()
    assert md["type"] == "dataset"
    assert md["table_name"] == table_name
    assert md["dataset_name"] == dataset_name
    assert md.get("model_label") == None
    assert md.get("model_revision") == None


def test_table_uri():
    table_name = "mytable"
    uri = LhURI.get_table_uri(table_name=table_name)
    assert uri.endswith(f"tables/{table_name}")
    assert table_name in uri, "Table name did not get included"
    parse = urlparse(uri)
    lh = LhURI(parse)
    assert lh.get_lh_table_name() == table_name
    md = lh.get_metadata()
    assert md["type"] == "table"
    assert md["table_name"] == table_name
    assert md.get("model_label") == None
    assert md.get("model_revision") == None
    assert md.get("dataset_name") == None


def test_fileset_uri():
    table_name = "mytable"
    name = "myfileset"
    revision = "123"
    uri = LhURI.get_fileset_uri(table_name=table_name, fileset_label=name, fileset_version=revision)
    assert uri.endswith(f"filesets/{table_name}/{name}/{revision}")
    assert table_name in uri, "Table name did not get included"
    assert name in uri, "Name did not get included"
    assert revision in uri, "Revision did not get included"
    parse = urlparse(uri)
    lh = LhURI(parse)
    assert lh.get_lh_table_name() == table_name
    assert lh.get_lh_fileset_label() == name
    assert lh.get_lh_fileset_version() == revision
    md = lh.get_metadata()
    assert md["type"] == "fileset"
    assert md["table_name"] == table_name
    assert md["fileset_label"] == name
    assert md["fileset_version"] == revision
    assert md.get("dataset_name") == None


def test_fileset_uri_no_revision():
    namespace = "mynamespace"
    table_name = "mytable"
    name = "myfileset"
    revision = ""
    uri = LhURI.get_fileset_uri(table_name=table_name, fileset_label=name)
    assert uri.endswith(f"filesets/{table_name}/{name}/{DEFAULT_FILESET_VERSION}")
    assert table_name in uri, "Table name did not get included"
    assert name in uri, "Name did not get included"
    uri = f"lh://somehost/{namespace}/filesets/{table_name}/{name}/"
    parse = urlparse(uri)
    lh = LhURI(parse)
    assert lh.get_lh_table_name() == table_name
    assert lh.get_lh_fileset_label() == name
    assert lh.get_lh_fileset_version() == DEFAULT_FILESET_VERSION
    md = lh.get_metadata()
    assert md["type"] == "fileset"
    assert md["table_name"] == table_name
    assert md["fileset_label"] == name
    assert md["fileset_version"] == DEFAULT_FILESET_VERSION
    assert md.get("dataset_name") is None


def test_add_uri_revision():
    namespace = "mynamespace"
    table_name = "mytable"
    name = "myfileset"
    revision = ""
    uri = f"lh://anything/{namespace}/filesets/{table_name}/{name}"
    lh = URI.get_uri(uri)
    assert isinstance(lh, LhURI)
    assert lh.get_lh_fileset_version() == DEFAULT_FILESET_VERSION
    uristr = lh.get_uristr(lh)
    assert f"{uristr}" == f"{uri}/{DEFAULT_FILESET_VERSION}"

    uri = f"lh://anything/{namespace}/models/{table_name}/{name}/"  # include trailing /
    lh = URI.get_uri(uri)
    assert isinstance(lh, LhURI)
    assert lh.get_lh_model_revision() == DEFAULT_MODEL_REVISION
    uristr = lh.get_uristr(lh)
    assert f"{uristr}" == f"{uri}{DEFAULT_MODEL_REVISION}"
