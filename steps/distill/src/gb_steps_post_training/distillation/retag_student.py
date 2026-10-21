#!/usr/bin/env python3
"""
Retag a Granite student onto a teacher's tokenizer, transplanting special-token
embeddings by structural role instead of mean-initializing them.

Ported from an earlier exploratory version of this same surgery. Every path, id, and
mapping here is derived or passed in rather than hardcoded to one sandbox, because
the earlier version held three absolute out-of-tree paths, three hardcoded token
ids, and a preference for a directory nobody outside that sandbox could read.

WHY THE TRANSPLANT EXISTS. A Granite base student and a Granite 4.2 teacher share
100,352 vocab entries and 96 added tokens, but disagree on **17 of those 96 ids**
(measured for both `4.1-3b-base -> 4.2-30b` and `4.0-350m-base -> 4.2-3b`;
identical disagreement sets). Two of the 17 are load-bearing on every single turn:

    id 100256   student <|pad|>          teacher <|im_start|>
    id 100257   student <|end_of_text|>  teacher <|im_end|>   (= EOS)

Training on ChatML data without retagging does not error. It silently treats the
turn markers as ordinary text. Retagging and then *mean-initializing* those rows
is worse in a specific way: under `tie_word_embeddings=True` the embedding matrix
is also the output projection, so a blank EOS row means the model cannot predict
end-of-turn and generates until it hits the length cap. That is exactly what retag
v1 did, and the pre-distillation model emitted gibberish as a result.

So each new ChatML control token is seeded from the original model's embedding for
the token that played the *same structural role* — turn-open from turn-open,
end-of-sequence from end-of-sequence. Only genuinely roleless tokens (a few
`<|unused_*|>`) fall back to the mean of the trained rows.

WHAT ELSE THIS WRITES, AND WHY IT MATTERS. The output directory gets a chat
template installed into it. `gold/` has no config knob for one: `sft.py` reads it
off the tokenizer, and puts `getattr(tokenizer, "chat_template")` into the dataset
cache key (`sft.py:1092`). A student directory with no template means
`apply_chat_template` has nothing to apply, and a template without
`{% generation %}` markers means `assistant_masks` comes back all-zero and
`sft.py:909` raises. Pass `--chat-template
templates/chatml_granite_42_generation.jinja` — the teacher's own upstream
template plus generation markers.

    python -m gb_steps_post_training.distillation.retag_student \
        --student <src model dir> --teacher <chatml tokenizer donor> \
        --out <dest> --chat-template templates/chatml_granite_42_generation.jinja \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from gb_steps_post_training.distillation import tokenizer_identity

# Resolved to ids at runtime from each tokenizer's own tables, never hardcoded:
# the same role sits at different ids in different checkpoint families.
#
#   new ChatML token  ->  original token whose embedding seeds it
TRANSPLANT_BY_ROLE = {
    "<|im_start|>": "<|start_of_role|>",   # turn open
    "<|im_end|>": "<|end_of_text|>",       # turn end / EOS -- the critical one
    "<s>": "<|end_of_text|>",              # sequence begin
    "</s>": "<|end_of_text|>",             # sequence end
    "<unk>": "<|unk|>",
    "[INST]": "<|start_of_role|>",         # instruction open
    "[/INST]": "<|end_of_role|>",          # instruction close
    "[AVAILABLE_TOOLS]": "<tools>",
    "[/AVAILABLE_TOOLS]": "</tools>",
    "[TOOL_RESULTS]": "<tool_response>",
    "[/TOOL_RESULTS]": "</tool_response>",
    "[TOOL_CALLS]": "<tool_call>",
}

EMB_KEY = "model.embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json")


def id_to_token(model_dir: str) -> dict[int, str]:
    with open(os.path.join(model_dir, "tokenizer.json")) as fh:
        t = json.load(fh)
    m = {v: k for k, v in t["model"]["vocab"].items()}
    for a in t.get("added_tokens", []):
        m[a["id"]] = a["content"]
    return m


def load_config(model_dir: str) -> dict:
    """Read a MODEL directory's config.json.

    Fails with an instruction rather than a traceback when the directory has none,
    because there is exactly one plausible way to get here wrong and it is worth
    naming: passing build_overlay's tokenizer OVERLAY as --teacher. That is a
    natural mistake -- the overlay is what distill-gold-train's
    teacher_tokenizer_path wants, and the two paths sit next to each other in a
    recipe -- and it has happened for real.

    An overlay is the wrong input here, and NOT because of the tokenizer_class
    trap. This module is immune to that by construction: it reads tokenizer.json
    as raw JSON (never through AutoTokenizer) and pins tokenizer_class on its own
    output. It is the wrong input because it is a tokenizer, and this function
    needs two facts that only a model config carries -- vocab_size, and the
    bos/eos/pad id scheme (see special_ids_from).
    """
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"no config.json in {model_dir}\n"
            f"  This must be a MODEL directory. If you passed a tokenizer overlay "
            f"(build_overlay's output), pass the teacher/student MODEL directory "
            f"instead -- retagging needs vocab_size and the bos/eos/pad ids, which "
            f"an overlay deliberately does not carry."
        )
    with open(path) as fh:
        return json.load(fh)


def shard_map(model_dir: str) -> dict[str, str]:
    """Weight name -> shard filename, for both sharded and single-file layouts."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as fh:
            return json.load(fh)["weight_map"]
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise SystemExit(f"no safetensors weights found in {model_dir}")


def build_plan(student_dir: str, teacher_dir: str) -> dict:
    """Classify every teacher id into reuse / move / transplant / mean-init.

    Resolved entirely from the two tokenizer.json files -- no weights loaded --
    so --dry-run is cheap and the plan is reviewable before any surgery.
    """
    S = id_to_token(student_dir)
    T = id_to_token(teacher_dir)
    s_tok2id = {tok: i for i, tok in S.items()}

    reuse, move, transplant, meaninit = [], [], [], []
    for nid in sorted(T):
        ntok = T[nid]
        oid = s_tok2id.get(ntok)
        if oid == nid:
            reuse.append(nid)
        elif oid is not None:
            # same surface string, different id: carry the trained row across
            move.append((nid, oid, ntok))
        elif ntok in TRANSPLANT_BY_ROLE:
            src = TRANSPLANT_BY_ROLE[ntok]
            if src not in s_tok2id:
                raise SystemExit(
                    f"transplant source {src!r} for {ntok!r} is absent from the "
                    f"student tokenizer. The role map does not fit this "
                    f"checkpoint pair; extend TRANSPLANT_BY_ROLE deliberately "
                    f"rather than letting a turn marker fall back to mean-init."
                )
            transplant.append((nid, s_tok2id[src], ntok, src))
        else:
            meaninit.append((nid, ntok))

    total = len(reuse) + len(move) + len(transplant) + len(meaninit)
    if total != len(T):
        raise SystemExit(f"id accounting: {total} classified vs {len(T)} teacher ids")
    return dict(reuse=reuse, move=move, transplant=transplant, meaninit=meaninit,
                n_student=len(S), n_teacher=len(T))


def special_ids_from(teacher_dir: str) -> dict[str, int]:
    """Take bos/eos/pad from the teacher's config, not from constants.

    retag_student_v2.py hardcoded 100283/100257/100257. Those happen to be right
    for the 4.2 teachers, and would be silently wrong for any other donor.
    """
    cfg = load_config(teacher_dir)
    out = {}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if cfg.get(key) is not None:
            out[key] = cfg[key]
    missing = {"bos_token_id", "eos_token_id", "pad_token_id"} - set(out)
    if missing:
        raise SystemExit(f"teacher config.json lacks {sorted(missing)}; "
                         f"cannot derive the student's special ids")
    return out


def place(src: str, dst: str, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass  # different filesystem, or a cache blob we cannot link
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True,
                    help="source model dir (weights + its own tokenizer)")
    ap.add_argument("--teacher", required=True,
                    help="tokenizer donor: the ChatML tokenizer to adopt")
    ap.add_argument("--out", required=True, help="destination model dir")
    ap.add_argument("--chat-template",
                    help="jinja file installed as <out>/chat_template.jinja. "
                         "Required unless --no-chat-template: gold/sft.py has no "
                         "config knob for one and reads it off the tokenizer.")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="skip template installation (the resulting dir cannot "
                         "train under gold/sft.py; for inspection only)")
    ap.add_argument("--tokenizer-identity", default="",
                    help="name recorded in tokenizer_identity.json and compared against "
                         "the corpus manifest by distill-gold-train. Defaults to the "
                         "teacher directory's basename, which is correct whenever that "
                         "basename identifies the tokenizer.")
    ap.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy",
                    help="how to place unmodified shards. hardlink is instant and "
                         "free but shares inodes with the source, so only use it "
                         "where nothing rewrites weights in place (default: copy)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and write nothing")
    args = ap.parse_args()

    if not args.chat_template and not args.no_chat_template:
        raise SystemExit("pass --chat-template <file>, or --no-chat-template to "
                         "acknowledge the output cannot train")
    if args.chat_template and not os.path.exists(args.chat_template):
        raise SystemExit(f"--chat-template not found: {args.chat_template}")

    s_cfg, t_cfg = load_config(args.student), load_config(args.teacher)
    if s_cfg["vocab_size"] != t_cfg["vocab_size"]:
        raise SystemExit(
            f"vocab_size differs: student {s_cfg['vocab_size']} vs teacher "
            f"{t_cfg['vocab_size']}. This script retags an equal-sized id space; "
            f"resizing embeddings is a different operation."
        )

    plan = build_plan(args.student, args.teacher)
    specials = special_ids_from(args.teacher)
    tied = bool(s_cfg.get("tie_word_embeddings"))

    # Is there anything to retag at all? Read off the plan rather than by
    # re-reading the two tokenizer.json files: build_plan has already classified
    # every teacher id, and already raises when the accounting does not add up, so
    # this cannot disagree with the surgery that follows.
    identical_vocab = (
        not plan["move"]
        and not plan["transplant"]
        and not plan["meaninit"]
        and plan["n_student"] == plan["n_teacher"] == len(plan["reuse"])
    )

    print(f"student : {args.student}")
    print(f"teacher : {args.teacher}")
    print(f"out     : {args.out}")
    print(f"vocab   : {s_cfg['vocab_size']}  tie_word_embeddings={tied}")
    print(f"specials from teacher: {specials}")
    print(f"\nreuse_in_place={len(plan['reuse'])}  moved={len(plan['move'])}  "
          f"transplant={len(plan['transplant'])}  mean_init={len(plan['meaninit'])}")
    for nid, oid, tok in plan["move"]:
        print(f"  MOVE       new {nid} <- old {oid}  {tok!r}")
    for nid, oid, ntok, otok in plan["transplant"]:
        print(f"  TRANSPLANT new {nid} {ntok!r} <- old {oid} {otok!r}")
    for nid, tok in plan["meaninit"]:
        print(f"  MEAN-INIT  new {nid}  {tok!r}")

    # The two rows that decide whether the retagged model can hold a conversation
    # at all. Loud, because a mean-init here is the v1 gibberish failure and it is
    # invisible in a loss curve.
    t_tokens = id_to_token(args.teacher)
    t_ids = {tok: i for i, tok in t_tokens.items()}
    critical = [("EOS", specials["eos_token_id"])]
    # The turn-open marker is looked up by content, not assumed to be at a fixed
    # id, and its absence is itself an error: a ChatML donor without <|im_start|>
    # is not a ChatML donor -- UNLESS the two vocabularies are the same, in which
    # case there is no role map to apply and nothing for it to be wrong about.
    if "<|im_start|>" in t_ids:
        critical.append(("turn-open", t_ids["<|im_start|>"]))
    elif identical_vocab:
        # Same vocabulary, id for id. Stated rather than passed over in silence,
        # because "no turn marker was checked" is the kind of thing that should be
        # visible in a log when someone later asks why a run masked the wrong span.
        # The boundary is still asserted, just elsewhere: the caller derives the
        # masking contract from the chat template this run installs.
        print("teacher tokenizer has no <|im_start|> and is not a ChatML donor, "
              "but its vocabulary is IDENTICAL to the student's "
              f"({plan['n_teacher']} ids, all reuse) -- nothing to retag, so the "
              "role map is vacuous rather than inapplicable")
    else:
        raise SystemExit("teacher tokenizer has no <|im_start|>; not a ChatML "
                         "donor, so the role map below does not apply")
    for role, nid in critical:
        kind = ("transplant" if any(t[0] == nid for t in plan["transplant"])
                else "move" if any(m[0] == nid for m in plan["move"])
                else "reuse" if nid in plan["reuse"] else "MEAN-INIT")
        print(f"\n  {role} id {nid} ({t_tokens.get(nid)!r}) is handled by: {kind}")
        if kind == "MEAN-INIT":
            raise SystemExit(
                f"{role} row {nid} would be mean-initialized. Under "
                f"tie_word_embeddings={tied} that row is also the output "
                f"projection; the model would be unable to emit it. Refusing."
            )

    weights = shard_map(args.student)
    if EMB_KEY not in weights:
        raise SystemExit(f"{EMB_KEY} absent from {args.student}")
    emb_shard = weights[EMB_KEY]
    head_shard = weights.get(LM_HEAD_KEY)
    print(f"\nembedding tensor in shard: {emb_shard}")
    if head_shard:
        print(f"lm_head tensor in shard  : {head_shard}  (untied -- retagged too)")
    elif not tied:
        print("NOTE: tie_word_embeddings is False but no lm_head.weight found; "
              "only the embedding is retagged.")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    import torch
    from safetensors.torch import load_file, save_file

    os.makedirs(args.out, exist_ok=True)

    def retag(tensor: "torch.Tensor") -> "torch.Tensor":
        orig = tensor.clone()
        new = tensor.clone()
        # Mean over rows the student actually trained, i.e. the ones whose id and
        # content both survive unchanged -- not over the whole matrix, which would
        # fold in the very untrained control rows we are trying to replace.
        mean_vec = orig.index_select(
            0, torch.tensor(plan["reuse"], dtype=torch.long)).mean(dim=0)
        for nid, oid, _tok in plan["move"]:
            new[nid] = orig[oid]
        for nid, oid, _ntok, _otok in plan["transplant"]:
            new[nid] = orig[oid]
        for nid, _tok in plan["meaninit"]:
            new[nid] = mean_vec
        return new

    touched = {emb_shard}
    tensors = load_file(os.path.join(args.student, emb_shard))
    tensors[EMB_KEY] = retag(tensors[EMB_KEY])
    if head_shard == emb_shard:
        tensors[LM_HEAD_KEY] = retag(tensors[LM_HEAD_KEY])
    save_file(tensors, os.path.join(args.out, emb_shard), metadata={"format": "pt"})
    print(f"wrote {emb_shard}")

    if head_shard and head_shard != emb_shard:
        t2 = load_file(os.path.join(args.student, head_shard))
        t2[LM_HEAD_KEY] = retag(t2[LM_HEAD_KEY])
        save_file(t2, os.path.join(args.out, head_shard), metadata={"format": "pt"})
        touched.add(head_shard)
        print(f"wrote {head_shard}")

    for sh in sorted(set(weights.values()) - touched):
        place(os.path.join(args.student, sh), os.path.join(args.out, sh),
              args.copy_mode)
    print(f"placed {len(set(weights.values()) - touched)} unmodified shard(s) "
          f"({args.copy_mode})")

    index = os.path.join(args.student, "model.safetensors.index.json")
    if os.path.exists(index):
        shutil.copy2(index, os.path.join(args.out, "model.safetensors.index.json"))

    # config + generation_config carry the teacher's special-id scheme
    s_cfg.update(specials)
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(s_cfg, fh, indent=2)
    gen_src = os.path.join(args.student, "generation_config.json")
    gen = {}
    if os.path.exists(gen_src):
        with open(gen_src) as fh:
            gen = json.load(fh)
    gen.update(specials)
    with open(os.path.join(args.out, "generation_config.json"), "w") as fh:
        json.dump(gen, fh, indent=2)

    # Adopt the teacher's tokenizer. TOKENIZER_FILES deliberately excludes the
    # vocab.json / merges.txt sidecars, but note that exclusion is NOT what makes
    # this correct -- under this tree's transformers 5.8.0 those sidecars are inert
    # (measured directly). Two things actually protect this output, and both
    # are below rather than here:
    #   1. tokenizer_class is rewritten to "PreTrainedTokenizerFast" a few lines
    #      down, which is what stops AutoTokenizer from constructing a class that
    #      imposes its own pre_tokenizer over the one in tokenizer.json.
    #   2. the teacher's own tokenizer.json pre_tokenizer is a plain ByteLevel, so
    #      even the GPT2Tokenizer override would be a no-op on this particular
    #      file. (It is emphatically NOT a no-op on granite-4.1-3b-base, whose
    #      pre_tokenizer is Sequence[Split(regex), ByteLevel] -- that asymmetry is
    #      why the base student needs an overlay and the teacher does not.)
    # Stakes if both were missing: 26.1 vs 3.29 PPL/token.
    # See docs/tokenizer_mismatch.md.
    for name in TOKENIZER_FILES:
        src = os.path.join(args.teacher, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, name))
    tc_path = os.path.join(args.out, "tokenizer_config.json")
    with open(tc_path) as fh:
        tc = json.load(fh)
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc.pop("chat_template", None)  # the file is canonical in transformers 5.x
    with open(tc_path, "w") as fh:
        json.dump(tc, fh, indent=2)

    if args.chat_template:
        shutil.copy2(args.chat_template,
                     os.path.join(args.out, "chat_template.jinja"))
        print(f"installed chat_template.jinja from {args.chat_template}")

    manifest = {
        "student": os.path.abspath(args.student),
        "teacher": os.path.abspath(args.teacher),
        "chat_template": (os.path.abspath(args.chat_template)
                          if args.chat_template else None),
        "vocab_size": s_cfg["vocab_size"],
        "tie_word_embeddings": tied,
        "special_ids": specials,
        "counts": {k: len(plan[k]) for k in ("reuse", "move", "transplant", "meaninit")},
        "move": [{"new": n, "old": o, "token": t} for n, o, t in plan["move"]],
        "transplant": [{"new": n, "old": o, "new_token": nt, "old_token": ot}
                       for n, o, nt, ot in plan["transplant"]],
        "mean_init": [{"id": n, "token": t} for n, t in plan["meaninit"]],
    }
    with open(os.path.join(args.out, "retag_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Name the tokenizer this student now carries. THIS FILE IS WHAT MAKES THE
    # ONE-TOKENIZER-PER-RUN GUARD REAL. distill-gold-train already implemented the
    # comparison (render_gold_config.validate) and read it from exactly this path -- but
    # nothing in the repo wrote it, so the read returned None and the guard was skipped on
    # every run it was supposed to protect. An assertion that cannot fail is worse than
    # none, because the plan doc counted it as done.
    #
    # The identity is the TEACHER's basename, not the student's and not a name of its own:
    # the retag copies the teacher's tokenizer.json byte-for-byte, so this student and the
    # teacher's overlay carry the same tokenizer and must compare EQUAL. --tokenizer-identity
    # overrides it for the case where two distinct teachers share a directory name.
    identity = args.tokenizer_identity or tokenizer_identity.derive_name(args.teacher)
    ident_file = tokenizer_identity.write(
        args.out, identity,
        produced_by="distill-tokenizer-align/retag_student",
        student=os.path.abspath(args.student),
        teacher=os.path.abspath(args.teacher),
    )
    print(f"recorded tokenizer identity {identity!r} -> {ident_file}")

    print(f"\nDONE -> {args.out}")
    print("Verify before training: load with "
          "PreTrainedTokenizerFast(tokenizer_file=...), render one example with "
          "return_assistant_tokens_mask=True, and confirm the mask is non-empty.")


if __name__ == "__main__":
    sys.exit(main())
