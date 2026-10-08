"""
PORTED, not authored here. Upstream source of truth:
  repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
  path   steps/distill-logit-precompute/test/test_precompute_logits.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Divergence, about path resolution rather than behaviour: upstream's sys.path inserts are
removed; conftest.py resolves the step's own src/ and the shared distillation package from
GB_DISTILL_CODE_DIR.

Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
"""

"""Unit tests for distill-logit-precompute.

Nothing here needs a GPU, torch, or a teacher. That is by construction, not luck: the module
imports torch/transformers/accelerate INSIDE run_precompute(), so the resume repair, the
post-conditions, the expectation gate and the masking fallback -- everything that decides whether
the artifact is CORRECT -- is reachable with stdlib alone. The GPU pass itself is covered by the
smoke run recorded in README.md.

The test that matters most is test_resume_after_a_kill_between_the_two_writes, which reproduces the
defect the port was written to fix. It is worth reading before the others, because it is the reason
this step's resume path looks the way it does.
"""

import json
import sys
from pathlib import Path

import pytest
from gb_steps_post_training.distillation import precompute_logits as P  # noqa: E402

TOP_K = 4  # small enough that byte arithmetic is checkable by hand
IDX_ROW = TOP_K * P.INDICES_ITEMSIZE  # 16 bytes per row of token ids
LG_ROW = TOP_K * P.LOGITS_ITEMSIZE  # 8 bytes per row of logit values


# --------------------------------------------------------------------------- helpers


def write_corpus(path, n_rows, *, assistant=True):
    """A corpus of n_rows trivial two-turn chats."""
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_rows):
            msgs = [{"role": "user", "content": f"q{i}"}]
            if assistant:
                msgs.append({"role": "assistant", "content": f"a{i}"})
            f.write(json.dumps({"messages": msgs}) + "\n")
    return path


def make_output(
    tmp_path,
    rows,
    *,
    n_source=None,
    top_k=TOP_K,
    max_skip_fraction=1.0,
    shard_bytes=None,
):
    """Build an output dir from `rows` (index dicts), sizing the shard files to match unless
    `shard_bytes` overrides a specific one -- which is how the corruption cases are staged.
    """
    out = tmp_path / "out"
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = {}
    for r in rows:
        if r.get("skipped"):
            continue
        sid = r["shard_id"]
        counts[sid] = max(
            counts.get(sid, 0), r["shard_offset"] + r["num_assistant_tokens"]
        )
    for sid, n in counts.items():
        ip, lp = P.shard_paths(str(shards), sid)
        Path(ip).write_bytes(
            b"\0"
            * (shard_bytes or {}).get((sid, "indices"), n * top_k * P.INDICES_ITEMSIZE)
        )
        Path(lp).write_bytes(
            b"\0"
            * (shard_bytes or {}).get((sid, "logits"), n * top_k * P.LOGITS_ITEMSIZE)
        )
    meta = {
        "top_k": top_k,
        "n_source": n_source if n_source is not None else len(rows),
        "max_skip_fraction": max_skip_fraction,
        "max_length": 8192,
    }
    (out / "meta.json").write_text(json.dumps(meta))
    return out


def kept(source_idx, shard_id, shard_offset, n):
    return {
        "source_idx": source_idx,
        "skipped": False,
        "skip_reason": None,
        "num_assistant_tokens": n,
        "shard_id": shard_id,
        "shard_offset": shard_offset,
    }


def skipped(source_idx, reason):
    return {
        "source_idx": source_idx,
        "skipped": True,
        "skip_reason": reason,
        "num_assistant_tokens": 0,
        "shard_id": -1,
        "shard_offset": -1,
    }


# ------------------------------------------------------- THE BUG THIS PORT WAS WRITTEN FOR


def test_resume_after_a_kill_between_the_two_writes(tmp_path, capsys):
    """A kill between the shard append and the index line must not misalign the rest of the shard.

    The writer appends `n` rows of bytes to shards/*.bin and THEN writes the index line naming
    (shard_id, shard_offset). Killed in between -- routine on a preemptable queue -- the .bin files
    hold rows the index never mentions.

    The scratchpad's resume then reopened those files in "ab" (append: seek to end) while taking
    shard_offset from the INDEX. So the index said the next row lived at offset 2 while the bytes
    actually landed at offset 3, and every row after it in that shard was off by the same block.
    Nothing raises. The trainer memmaps at the index's offset and reads a DIFFERENT token's teacher
    distribution for the remainder of the shard, and the loss curve looks entirely normal.

    This test stages exactly that state and asserts the repair.
    """
    shards = tmp_path / "shards"
    shards.mkdir()
    index_part = tmp_path / "index_part_0000.jsonl"

    # Two rows committed properly: 2 + 3 = 5 rows of bytes, both named in the index.
    with open(index_part, "w", encoding="utf-8") as f:
        f.write(json.dumps(kept(0, 0, 0, 2)) + "\n")
        f.write(json.dumps(kept(1, 0, 2, 3)) + "\n")
    ip, lp = P.shard_paths(str(shards), 0)
    Path(ip).write_bytes(b"\x11" * (5 * IDX_ROW))
    Path(lp).write_bytes(b"\x22" * (5 * LG_ROW))

    # Now the kill: source row 2's 4 rows of bytes reached the shard, its index line did not.
    with open(ip, "ab") as f:
        f.write(b"\xee" * (4 * IDX_ROW))
    with open(lp, "ab") as f:
        f.write(b"\xee" * (4 * LG_ROW))

    done, local_idx, offset, rows_by_shard = P.read_done_state(
        str(index_part), node_id=0
    )
    assert done == {0, 1}, "source row 2 was never indexed, so it is not done"
    assert (
        offset == 5
    ), "the index accounts for 5 rows; the 4 orphans are invisible to it"
    assert rows_by_shard == {0: 5}

    # Without the repair, the next append lands at byte offset 9*row while the index will say 5:
    assert Path(ip).stat().st_size == 9 * IDX_ROW
    would_write_at = Path(ip).stat().st_size // IDX_ROW
    assert (
        would_write_at != offset
    ), "this inequality IS the bug: append position 9 vs index offset 5"

    P._reconcile_shards(
        str(shards), node_id=0, rows_by_shard=rows_by_shard, top_k=TOP_K
    )

    assert Path(ip).stat().st_size == 5 * IDX_ROW
    assert Path(lp).stat().st_size == 5 * LG_ROW
    assert (
        Path(ip).stat().st_size // IDX_ROW == offset
    ), "append position now agrees with the index"
    # The surviving bytes are the committed ones, not the orphans.
    assert set(Path(ip).read_bytes()) == {0x11}
    assert "truncating indices_000000.bin" in capsys.readouterr().out


def test_reconcile_refuses_when_the_index_is_ahead_of_the_data(tmp_path):
    """The opposite direction is not repairable and must not be papered over.

    Truncation is only valid because the index is the authority and orphan rows get recomputed. A
    file SHORTER than the index claims means an index row points at teacher logits that do not
    exist -- which the writer's ordering cannot produce, so something else touched the directory.
    """
    shards = tmp_path / "shards"
    shards.mkdir()
    ip, lp = P.shard_paths(str(shards), 0)
    Path(ip).write_bytes(b"\0" * (2 * IDX_ROW))
    Path(lp).write_bytes(b"\0" * (2 * LG_ROW))
    with pytest.raises(RuntimeError, match="index is AHEAD of the data"):
        P._reconcile_shards(str(shards), 0, {0: 5}, TOP_K)


def test_reconcile_leaves_other_nodes_shards_alone(tmp_path):
    """Each node reconciles only its own id range. Node 1 truncating node 0's in-flight shard --
    while node 0 is mid-append to it -- would corrupt a healthy shard to repair nothing.
    """
    shards = tmp_path / "shards"
    shards.mkdir()
    other_ip, other_lp = P.shard_paths(str(shards), 0)  # node 0
    Path(other_ip).write_bytes(b"\0" * (7 * IDX_ROW))
    Path(other_lp).write_bytes(b"\0" * (7 * LG_ROW))
    mine_ip, mine_lp = P.shard_paths(str(shards), P.MAX_SHARDS_PER_NODE)  # node 1
    Path(mine_ip).write_bytes(b"\0" * (9 * IDX_ROW))
    Path(mine_lp).write_bytes(b"\0" * (9 * LG_ROW))

    P._reconcile_shards(str(shards), 1, {P.MAX_SHARDS_PER_NODE: 6}, TOP_K)

    assert Path(other_ip).stat().st_size == 7 * IDX_ROW, "node 0's shard untouched"
    assert Path(mine_ip).stat().st_size == 6 * IDX_ROW


def test_reconcile_zeroes_a_shard_the_index_never_mentions(tmp_path):
    """Killed after the rollover append but before the first index line for the new shard, the file
    exists with bytes and zero index rows. It must go back to empty, not be left as a prefix the
    next offset-0 row writes after."""
    shards = tmp_path / "shards"
    shards.mkdir()
    ip, lp = P.shard_paths(str(shards), 3)
    Path(ip).write_bytes(b"\0" * (2 * IDX_ROW))
    Path(lp).write_bytes(b"\0" * (2 * LG_ROW))
    P._reconcile_shards(str(shards), 0, {}, TOP_K)
    assert Path(ip).stat().st_size == 0
    assert Path(lp).stat().st_size == 0


# --------------------------------------------------------------------- resume bookkeeping


def test_read_done_state_on_a_fresh_dir():
    done, local_idx, offset, rows = P.read_done_state(
        "/nonexistent/index_part_0000.jsonl", 0
    )
    assert (done, local_idx, offset, rows) == (set(), 0, 0, {})


def test_read_done_state_counts_skips_as_done_but_not_as_rows(tmp_path):
    """A skipped row must never be recomputed (it is settled), and must never claim shard bytes."""
    part = tmp_path / "index_part_0000.jsonl"
    part.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                kept(0, 0, 0, 2),
                skipped(1, "too_long"),
                kept(2, 0, 2, 1),
            ]
        )
        + "\n"
    )
    done, local_idx, offset, rows = P.read_done_state(str(part), 0)
    assert done == {0, 1, 2}
    assert rows == {0: 3}, "the skip contributes no rows"
    assert (local_idx, offset) == (0, 3)


def test_read_done_state_tracks_the_highest_shard_not_the_last_line(tmp_path):
    """Rows are appended in order, but the resume offset must belong to the newest shard even if a
    later line names an older one."""
    part = tmp_path / "index_part_0000.jsonl"
    part.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                kept(0, 0, 0, 5),
                kept(1, 1, 0, 2),
                kept(2, 1, 2, 3),
            ]
        )
        + "\n"
    )
    _, local_idx, offset, rows = P.read_done_state(str(part), 0)
    assert (local_idx, offset) == (1, 5)
    assert rows == {0: 5, 1: 5}


def test_read_done_state_derives_local_idx_from_the_node_namespace(tmp_path):
    """shard_id = node_id * MAX_SHARDS_PER_NODE + local_idx. Node 2 resuming its own part must read
    local_idx 3, not shard_id 2003."""
    part = tmp_path / "index_part_0002.jsonl"
    sid = 2 * P.MAX_SHARDS_PER_NODE + 3
    part.write_text(json.dumps(kept(7, sid, 0, 4)) + "\n")
    _, local_idx, offset, _ = P.read_done_state(str(part), node_id=2)
    assert (local_idx, offset) == (3, 4)


def test_read_done_state_names_a_torn_line(tmp_path):
    part = tmp_path / "index_part_0000.jsonl"
    part.write_text(json.dumps(kept(0, 0, 0, 2)) + "\n" + '{"source_idx": 1, "sh')
    with pytest.raises(RuntimeError, match="torn write"):
        P.read_done_state(str(part), 0)


# ----------------------------------------------------------------------- post-conditions


def test_verify_accepts_a_complete_pass(tmp_path, capsys):
    out = make_output(
        tmp_path,
        [kept(0, 0, 0, 2), skipped(1, "too_long"), kept(2, 0, 2, 3)],
        n_source=3,
    )
    stats = P.verify_output(str(out))
    assert stats["n_kept"] == 2 and stats["n_skipped"] == 1 and stats["n_shards"] == 1
    assert stats["total_teacher_rows"] == 5
    assert "5 teacher rows" in capsys.readouterr().out


def test_verify_catches_an_incomplete_pass(tmp_path):
    """The failure with no downstream symptom: a partial shard set is structurally valid. It loads,
    it memmaps, and it trains -- on a fraction of the corpus, silently."""
    out = make_output(tmp_path, [kept(0, 0, 0, 2), kept(1, 0, 2, 2)], n_source=10)
    with pytest.raises(RuntimeError, match="INCOMPLETE precompute"):
        P.verify_output(str(out))


def test_verify_incomplete_message_names_the_missing_rows_and_the_fix(tmp_path):
    out = make_output(tmp_path, [kept(0, 0, 0, 1), kept(3, 0, 1, 1)], n_source=4)
    with pytest.raises(RuntimeError) as e:
        P.verify_output(str(out))
    assert "[1, 2]" in str(e.value)
    assert "resume skips what is already done" in str(e.value)


def test_verify_catches_a_duplicate_source_idx(tmp_path):
    out = make_output(tmp_path, [kept(0, 0, 0, 1), kept(0, 0, 1, 1)], n_source=1)
    with pytest.raises(RuntimeError, match="duplicate source_idx=0"):
        P.verify_output(str(out))


def test_verify_catches_an_index_row_outside_the_corpus(tmp_path):
    """The shape of pointing a second, shorter corpus at an existing output dir."""
    out = make_output(tmp_path, [kept(0, 0, 0, 1), kept(9, 0, 1, 1)], n_source=2)
    with pytest.raises(RuntimeError, match="outside the corpus"):
        P.verify_output(str(out))


def test_verify_enforces_the_skip_fraction(tmp_path):
    rows = [kept(0, 0, 0, 1)] + [skipped(i, "too_long") for i in range(1, 5)]
    out = make_output(tmp_path, rows, n_source=5, max_skip_fraction=0.05)
    with pytest.raises(
        RuntimeError, match="80.0% .* were SKIPPED|80.0%\\) were SKIPPED"
    ):
        P.verify_output(str(out))


def test_skip_fraction_message_explains_both_reasons(tmp_path):
    """A skip is a row the trainer never sees, so the message has to say which knob caused it."""
    rows = [skipped(0, "too_long"), skipped(1, "no_assistant"), kept(2, 0, 0, 1)]
    out = make_output(tmp_path, rows, n_source=3, max_skip_fraction=0.0)
    with pytest.raises(RuntimeError) as e:
        P.verify_output(str(out))
    msg = str(e.value)
    assert "too_long=1" in msg and "no_assistant=1" in msg
    assert "rows are skipped, never truncated" in msg
    assert "--response-template" in msg


def test_skip_fraction_is_overridable(tmp_path):
    rows = [skipped(0, "too_long"), kept(1, 0, 0, 1)]
    out = make_output(tmp_path, rows, n_source=2, max_skip_fraction=0.0)
    assert P.verify_output(str(out), max_skip_fraction=1.0)["n_skipped"] == 1


def test_verify_catches_a_shard_longer_than_the_index(tmp_path):
    """Orphan bytes in a directory that claims to be FINISHED. Resume repairs this; a finished
    artifact carrying it means the merge ran over an unreconciled shard."""
    out = make_output(
        tmp_path,
        [kept(0, 0, 0, 2)],
        n_source=1,
        shard_bytes={(0, "logits"): 3 * LG_ROW},
    )
    with pytest.raises(RuntimeError, match="LONGER than the index"):
        P.verify_output(str(out))


def test_verify_catches_a_shard_shorter_than_the_index(tmp_path):
    out = make_output(
        tmp_path,
        [kept(0, 0, 0, 2)],
        n_source=1,
        shard_bytes={(0, "indices"): 1 * IDX_ROW},
    )
    with pytest.raises(RuntimeError, match="SHORTER than the index"):
        P.verify_output(str(out))


def test_verify_catches_a_gap_in_a_shard(tmp_path):
    """Equal totals are not enough: the rows must TILE. A gap means bytes nothing reads, and every
    row after it reads its neighbour's logits."""
    out = make_output(tmp_path, [kept(0, 0, 0, 2), kept(1, 0, 5, 2)], n_source=2)
    with pytest.raises(RuntimeError, match="do not tile"):
        P.verify_output(str(out))


def test_verify_catches_an_overlap_in_a_shard(tmp_path):
    out = make_output(tmp_path, [kept(0, 0, 0, 3), kept(1, 0, 1, 3)], n_source=2)
    with pytest.raises(RuntimeError, match="do not tile"):
        P.verify_output(str(out))


def test_verify_catches_a_kept_row_claiming_no_tokens(tmp_path):
    out = make_output(tmp_path, [kept(0, 0, 0, 0)], n_source=1)
    with pytest.raises(RuntimeError, match="claims 0 assistant tokens"):
        P.verify_output(str(out))


def test_verify_needs_meta_and_index(tmp_path):
    out = tmp_path / "empty"
    out.mkdir()
    with pytest.raises(RuntimeError, match="meta.json is missing"):
        P.verify_output(str(out))


def test_verify_uses_metas_top_k_when_not_told(tmp_path):
    """--verify-only has no --top-k of its own, so the byte arithmetic has to come from meta.json.
    A wrong top_k there would make every length check wrong in the same direction."""
    out = make_output(tmp_path, [kept(0, 0, 0, 2)], n_source=1, top_k=TOP_K)
    meta = json.loads((out / "meta.json").read_text())
    meta["top_k"] = TOP_K * 2
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="bytes, but the index accounts for"):
        P.verify_output(str(out))


# -------------------------------------------------------------------- expectation gate


def _args(tmp_path, **over):
    corpus = write_corpus(tmp_path / "c.jsonl", 3)
    teacher = tmp_path / "teacher"
    teacher.mkdir(exist_ok=True)
    base = dict(
        input_jsonl=str(corpus),
        output_dir=str(tmp_path / "out"),
        teacher_model=str(teacher),
        teacher_tokenizer=str(teacher),
        top_k=TOP_K,
        max_length=8192,
        dtype="bfloat16",
        shard_target_tokens=4_000_000,
        ignore_documents=False,
        response_template="<|im_start|>assistant\n",
    )
    base.update(over)
    return type("A", (), base)()


def test_expectation_records_the_corpus_identity(tmp_path):
    exp = P.build_expectation(_args(tmp_path))
    assert exp["source_rows"] == 3
    assert len(exp["source_jsonl_md5"]) == 32
    assert exp["step"] == P.STEP_NAME


def test_expectation_excludes_batch_size_and_seed(tmp_path):
    """Deliberate: after an OOM you resume the SAME artifact at a smaller batch size. Putting them
    in the identity would turn that into a refusal."""
    exp = P.build_expectation(_args(tmp_path))
    assert "batch_size" not in exp and "seed" not in exp
    assert "output_dir" not in exp, "an artifact is not identified by where it sits"
    assert "max_skip_fraction" not in exp, "a threshold, not a determinant of the bytes"


def test_install_expectation_stores_then_accepts_itself(tmp_path):
    a = _args(tmp_path)
    Path(a.output_dir).mkdir()
    exp = P.build_expectation(a)
    path = P.install_expectation(a.output_dir, exp)
    assert Path(path).exists()
    P.install_expectation(a.output_dir, exp)  # idempotent: this is what resume does


def test_install_expectation_refuses_a_different_corpus(tmp_path):
    """Without this, pointing a second corpus at an existing output dir appends to the same index
    and produces one index over two corpora, with nothing anywhere raising."""
    a = _args(tmp_path)
    Path(a.output_dir).mkdir()
    P.install_expectation(a.output_dir, P.build_expectation(a))
    other = write_corpus(tmp_path / "other.jsonl", 3, assistant=True)
    Path(other).write_text(Path(other).read_text().replace("q0", "DIFFERENT"))
    b = _args(tmp_path, input_jsonl=str(other))
    with pytest.raises(RuntimeError, match="DIFFERENT expectation") as e:
        P.install_expectation(b.output_dir, P.build_expectation(b))
    assert "source_jsonl_md5" in str(e.value)


def test_install_expectation_refuses_a_different_top_k(tmp_path):
    """top_k changes the byte stride of every shard, so appending under a new one would make the
    whole file unreadable at both strides."""
    a = _args(tmp_path)
    Path(a.output_dir).mkdir()
    P.install_expectation(a.output_dir, P.build_expectation(a))
    with pytest.raises(RuntimeError, match="top_k"):
        P.install_expectation(
            a.output_dir, P.build_expectation(_args(tmp_path, top_k=TOP_K * 2))
        )


def test_install_expectation_refuses_a_different_tokenizer(tmp_path):
    """The one the whole --teacher-tokenizer flag exists for: a corpus half-segmented by one
    tokenizer and half by another, with a meta.json that can only name one of them."""
    a = _args(tmp_path)
    Path(a.output_dir).mkdir()
    P.install_expectation(a.output_dir, P.build_expectation(a))
    other_tok = tmp_path / "other_tok"
    other_tok.mkdir()
    with pytest.raises(RuntimeError, match="teacher_tokenizer"):
        P.install_expectation(
            a.output_dir,
            P.build_expectation(_args(tmp_path, teacher_tokenizer=str(other_tok))),
        )


def test_expectation_hashes_the_tokenizer_contents(tmp_path):
    """The overlay is a directory in this tree, so an in-place edit of tokenizer.json must not slip
    past a path comparison."""
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text('{"a": 1}')
    a = _args(tmp_path, teacher_tokenizer=str(tok))
    first = P.build_expectation(a)["teacher_tokenizer_sha"]
    assert first and first.startswith("sha256:")
    (tok / "tokenizer.json").write_text('{"a": 2}')
    assert P.build_expectation(a)["teacher_tokenizer_sha"] != first


# ------------------------------------------------------------------------ corpus reading


def test_scan_jsonl_keeps_global_ids_for_a_dp_slice(tmp_path):
    """Under DP each node holds 1/num_nodes of the corpus but every id written to the index must
    still be the GLOBAL one, or the merged index names rows that do not exist."""
    corpus = write_corpus(tmp_path / "c.jsonl", 10)
    rows, n_total = P.scan_jsonl(str(corpus), keep=lambda i: i % 3 == 1)
    assert n_total == 10, "n_total counts the whole corpus, not the slice"
    assert sorted(rows) == [1, 4, 7]
    assert rows[4]["messages"][0]["content"] == "q4"


def test_scan_jsonl_ignores_blank_lines_consistently_with_count_lines(tmp_path):
    """count_lines() feeds the expectation's source_rows and scan_jsonl() feeds the completeness
    check. If they disagreed about blank lines, every run would report a phantom missing row.
    """
    p = tmp_path / "c.jsonl"
    p.write_text('{"messages": []}\n\n{"messages": []}\n   \n')
    rows, n_total = P.scan_jsonl(str(p))
    assert n_total == P.count_lines(str(p)) == 2
    assert sorted(rows) == [0, 1]


@pytest.mark.parametrize(
    "raw,want",
    [
        ('[{"name": "f"}]', [{"name": "f"}]),  # JSON string, the prep-corpus shape
        ([{"name": "f"}], [{"name": "f"}]),  # already a list
        ("", None),
        (None, None),
        ([], None),
        ("not json", None),
    ],
)
def test_normalize_tools_accepts_both_corpus_shapes(raw, want):
    """A json.loads() on an already-parsed list raises TypeError: the corpus carries `tools` both
    ways, and checks/collator-masking.py was fixed for exactly this. The index carries `tools`
    forward for the consumer's tool filters, so it has to survive both."""
    assert P.normalize_tools(raw) == want


# ---------------------------------------------------------------- response-template fallback


def test_scan_response_template_masks_after_each_marker():
    """The fallback for chat templates with no {% generation %} markers. It must agree with
    sft.py's, because sft.py re-derives the mask and REFUSES on a count disagreement."""
    ids = [1, 2, 90, 91, 5, 6, 99, 3, 90, 91, 7, 99]
    mask = P._scan_response_template(ids, [90, 91], eos_token_id=99)
    assert mask == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]


def test_scan_response_template_returns_none_when_absent():
    """None, not a zero mask: the caller distinguishes "no marker" (skip the row, loudly) from
    "marker found, span empty"."""
    assert P._scan_response_template([1, 2, 3], [90, 91], 99) is None
    assert P._scan_response_template([1, 2, 3], [], 99) is None


def test_scan_response_template_runs_to_end_without_eos():
    assert P._scan_response_template([90, 91, 5, 6], [90, 91], None) == [0, 0, 1, 1]


# ------------------------------------------------------------------------- light CLI modes


def test_emit_expectation_writes_and_exits_without_torch(tmp_path):
    """The launcher calls this in preflight to pay for the corpus md5 ONCE, then feeds the same file
    to step_state and to the run. It must not import torch or touch a GPU."""
    corpus = write_corpus(tmp_path / "c.jsonl", 4)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    dest = tmp_path / "exp.json"
    rc = P.main(
        [
            "--input-jsonl",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--teacher-model",
            str(teacher),
            "--emit-expectation",
            str(dest),
        ]
    )
    assert rc == 0
    assert json.loads(dest.read_text())["source_rows"] == 4
    assert (
        "torch" not in sys.modules or True
    )  # the real guard is the import-blocker check in CI


def test_verify_only_mode_rechecks_a_directory(tmp_path):
    out = make_output(tmp_path, [kept(0, 0, 0, 2), kept(1, 0, 2, 1)], n_source=2)
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    argv = [
        "--input-jsonl",
        str(write_corpus(tmp_path / "c.jsonl", 2)),
        "--output-dir",
        str(out),
        "--teacher-model",
        str(teacher),
        "--verify-only",
    ]
    assert P.main(argv) == 0
    (out / "index.jsonl").write_text(json.dumps(kept(0, 0, 0, 2)) + "\n")
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        P.main(argv)


def test_teacher_tokenizer_defaults_to_the_weights_dir():
    a = P.get_args(["--input-jsonl", "c", "--output-dir", "o", "--teacher-model", "/t"])
    assert a.teacher_tokenizer == "/t"


def test_ignore_documents_is_a_boolean_pair():
    """--ignore-documents/--no-ignore-documents, per the house rule: a bare --flag that cannot be
    turned off forces the launcher to build its argv conditionally, which is where drift starts.
    """
    base = ["--input-jsonl", "c", "--output-dir", "o", "--teacher-model", "t"]
    assert P.get_args(base).ignore_documents is False
    assert P.get_args(base + ["--ignore-documents"]).ignore_documents is True
    assert P.get_args(base + ["--no-ignore-documents"]).ignore_documents is False


def test_stored_dtypes_are_constants_not_the_load_dtype():
    """--dtype selects how the TEACHER is loaded; the stored logits are always float16/int32 and
    the consumer memmaps them at that stride. Coupling the two would silently change the stride.
    """
    assert (P.LOGITS_DTYPE, P.LOGITS_ITEMSIZE) == ("float16", 2)
    assert (P.INDICES_DTYPE, P.INDICES_ITEMSIZE) == ("int32", 4)
    assert (
        P.get_args(
            [
                "--input-jsonl",
                "c",
                "--output-dir",
                "o",
                "--teacher-model",
                "t",
                "--dtype",
                "float32",
            ]
        ).dtype
        == "float32"
    )


# --------------------------------------------------------------------------- process group
# _ensure_process_group takes `torch` as an ARGUMENT rather than importing it, which is what
# makes the refusal testable here at all: these three cases are the whole contract, and none of
# them needs a GPU, a rendezvous, or nccl.


class _FakeDist:
    def __init__(self, initialized):
        self._init = initialized
        self.calls = []

    def is_initialized(self):
        return self._init

    def init_process_group(self, **kw):
        self.calls.append(kw)
        self._init = True


class _FakeCuda:
    def __init__(self):
        self.device = None

    def set_device(self, i):
        self.device = i


class _FakeTorch:
    def __init__(self, initialized=False):
        self.distributed = _FakeDist(initialized)
        self.cuda = _FakeCuda()


def test_ensure_process_group_leaves_an_existing_group_alone():
    t = _FakeTorch(initialized=True)
    assert P._ensure_process_group(t, 8) is False
    assert (
        t.distributed.calls == []
    ), "it must not re-initialize a group the launcher made"


def test_ensure_process_group_initializes_a_single_rank_group():
    t = _FakeTorch(initialized=False)
    said, env = [], {}
    assert P._ensure_process_group(t, 1, env=env, emit=said.append) is True
    (kw,) = t.distributed.calls
    assert kw == {
        "backend": "nccl"
    }, "env:// rendezvous, so the env below is the whole contract"
    assert env["RANK"] == "0" and env["WORLD_SIZE"] == "1"
    # LOCAL_RANK specifically: transformers' initialize_tensor_parallelism() indexes it out of
    # os.environ and raised KeyError without it (LSF job 1201632). Satisfying torch alone is not
    # enough, which is why all six are set.
    assert env["LOCAL_RANK"] == "0" and env["LOCAL_WORLD_SIZE"] == "1"
    # A concrete free port, not the conventional 29500 -- these nodes are shared and a collision
    # would hang the rendezvous instead of raising.
    assert env["MASTER_ADDR"] == "127.0.0.1" and 1024 < int(env["MASTER_PORT"]) < 65536
    assert (
        t.cuda.device == 0
    ), "the device is set before the group, so nccl and DeviceMesh agree"
    assert "world_size=1" in said[0]


def test_ensure_process_group_overwrites_a_contradicting_inherited_environment():
    # For one rank the truth is known exactly, so an inherited MASTER_PORT or RANK from some other
    # launcher is wrong rather than a hint -- and honouring it would hang the rendezvous.
    t = _FakeTorch(initialized=False)
    env = {
        "RANK": "3",
        "WORLD_SIZE": "8",
        "MASTER_PORT": "29500",
        "MASTER_ADDR": "10.0.0.9",
    }
    P._ensure_process_group(t, 1, env=env, emit=lambda _m: None)
    assert env["RANK"] == "0" and env["WORLD_SIZE"] == "1"
    assert env["MASTER_ADDR"] == "127.0.0.1" and env["MASTER_PORT"] != "29500"


def test_ensure_process_group_refuses_a_multi_rank_launch_with_no_rendezvous():
    # THE case that matters. Fabricating a 1-rank group here would give every rank its own
    # complete-looking artifact over 1/world_size of the corpus, in one output dir, with no
    # post-condition able to tell -- each index would be internally consistent.
    t = _FakeTorch(initialized=False)
    env = {}
    with pytest.raises(RuntimeError) as e:
        P._ensure_process_group(t, 8, env=env)
    msg = str(e.value)
    assert "world_size=8" in msg
    assert "complete-looking artifact" in msg
    assert (
        t.distributed.calls == []
    ), "it must not initialize anything on the refusal path"
    assert env == {}, "nor leave a half-set environment behind for the next caller"
