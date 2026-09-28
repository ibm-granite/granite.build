from gbserver.environment.io.descriptors import HfInputIO, InputIO


def test_hf_input_io_fields():
    io = HfInputIO(
        repo="ns/model",
        revision="main",
        type="model",
        dest="/cache/ns/model/main",
        token="tok",
    )
    assert isinstance(io, InputIO)
    assert io.repo == "ns/model"
    assert io.dest == "/cache/ns/model/main"
