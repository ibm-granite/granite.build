from gbserver.environment.io.descriptors import (
    HfInputIO,
    HfOutputIO,
    InlineDeferredPush,
    InputIO,
    OutputIO,
)


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


def test_hf_output_io_has_no_src_field():
    io = HfOutputIO(
        repo="ns/out",
        revision="main",
        private=True,
        resource_group_id=None,
        path_in_repo="",
        uri="hf://ns/out",
        binding_id="out",
        token="tok",
        hf_type="model",
    )
    assert isinstance(io, OutputIO)
    # src is resolved at runtime from the GB_ARTIFACT_PATH marker, not stored.
    assert not hasattr(io, "src")


def test_inline_deferred_push_is_distinct_sentinel():
    from gbserver.types.buildconfig import BuildTargetStepConfig

    sentinel = InlineDeferredPush()
    assert not isinstance(sentinel, BuildTargetStepConfig)
    assert isinstance(sentinel, InlineDeferredPush)
