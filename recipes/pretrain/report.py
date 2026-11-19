# SPDX-License-Identifier: Apache-2.0

"""Roll the pretrain recipe's eval results up into one table per rung.

Usage:

    report.py <root> [--out DIR]

<root> is the directory the eval targets write under -- EVAL_ROOT/EXP/RUN_NAME --
so the files joined here are:

    <root>/ckpt_<N>/<entry>/results.json

where <N> is the optimizer step the checkpoint was taken at (the rung) and <entry>
is the eval entry scored on it. Writes, into --out (default: <root> itself, so the
tables land beside the data they summarise and a hand run needs no second path):

* summary.csv -- one row per rung, one column per entry, the cell being that
  run's headline value. Rows are rungs because a rung is what the recipe adds over
  time: a later rung appends a row rather than rewriting the header, and the file
  reads top to bottom as training progresses. Cells are bare values or empty, so
  the file stays numerically parseable.
* per_task.csv -- long form: rung, entry, task, metric, value, n. Every entry
  also contributes its headline as a row with task (headline), so this file on
  its own carries every number and every n.
* summary.md -- the same headline table for people to read, plus the metric each
  entry is measured in and a Notes section listing every gap and oddity.

THE SCHEMA READ HERE is the one granite-evals writes for every benchmark:
write_results in steps/eval/src/core/granite_evals/results.py. Top level:
benchmark, metric (the headline metric's name), value (the headline
value), n, and details (per-benchmark). The headline value is therefore
read, never computed: value is the number the benchmark itself declares
headline, and metric names it.

PER-TASK DETAIL is per-benchmark, and three detail keys exist in this repo, each a
mapping of key to {metric: value, "n": n}: details.per_task
(steps/eval/nemoskills/src/nemo_skills_ruler.py, keys = RULER tasks, metric
accuracy) and details.per_language / details.per_subject (the lmeval
family, steps/eval/lmeval/src/chat_if.py summarize, metric exact_match).
All three are read; the language and subject keys are prefixed in the task column
so the two cannot be confused. n is copied from the file as it stands and is
never computed or inferred: in some harnesses it counts batches rather than
samples, and only the file knows which.

NOTHING HERE FAILS ON MISSING DATA. An eval target can report SUCCESS having
written no results at all, and a crashed one has been seen to leave another eval's
results.json behind, so a missing or unreadable file is an empty cell and a note,
and a record whose benchmark does not match its directory keeps its value (an
entry name need not be a benchmark id) but is flagged in stderr and in the notes.
That is also why the only argument is a plain path: when one eval fails the
combined target never runs, and this script still has to produce the table for the
rungs that did finish.

Standard library only, and no import from this repository: this file is shipped
inline, in a shell heredoc, into a bare python:3.12-slim container.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

RESULTS = "results.json"
HEADLINE = "(headline)"
# details key -> prefix for the task column. per_task needs none; the other two are
# prefixed so a language and a subject of the same name stay distinct.
SECTIONS = (
    ("per_task", ""),
    ("per_language", "language:"),
    ("per_subject", "subject:"),
)


def rung_key(name):
    """Sort key for a ckpt_<N> directory: numerically, so ckpt_10 follows ckpt_2."""
    suffix = name[len("ckpt_") :]
    if suffix.isdigit():
        return (0, int(suffix), "")
    return (1, 0, name)


def rung_label(name):
    suffix = name[len("ckpt_") :]
    return str(int(suffix)) if suffix.isdigit() else name


def cell(value):
    """A value as written, or empty. Numbers and strings are carried verbatim."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def read_record(path):
    """(record, error): the parsed results.json, or None and why not."""
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except FileNotFoundError:
        return None, "no " + RESULTS
    except (OSError, ValueError) as exc:
        return None, "unreadable " + RESULTS + ": " + str(exc)
    if not isinstance(record, dict):
        return None, "unreadable " + RESULTS + ": not a JSON object"
    return record, None


def task_rows(rung, entry, record):
    """The long-form rows for one record: its headline, then its per-task detail."""
    metric = cell(record.get("metric"))
    rows = [
        [
            rung,
            entry,
            HEADLINE,
            metric,
            cell(record.get("value")),
            cell(record.get("n")),
        ]
    ]
    details = record.get("details")
    if not isinstance(details, dict):
        return rows
    for key, prefix in SECTIONS:
        section = details.get(key)
        if not isinstance(section, dict):
            continue
        for task in sorted(section):
            scores = section[task]
            name = prefix + str(task)
            if not isinstance(scores, dict):
                # A bare number per task (no metric name of its own): the record's
                # own metric names it, and the file states no n for it.
                rows.append([rung, entry, name, metric, cell(scores), ""])
                continue
            count = cell(scores.get("n"))
            for inner in sorted(scores):
                if inner == "n":
                    continue
                value = scores[inner]
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    rows.append([rung, entry, name, str(inner), cell(value), count])
    return rows


def collect(root):
    """(rungs, entries, headline, per_task, notes) over every ckpt_<N>/<entry>."""
    rungs, entries, headline, rows, notes = [], set(), {}, [], []
    names = sorted((n for n in os.listdir(root) if n.startswith("ckpt_")), key=rung_key)
    for name in names:
        directory = os.path.join(root, name)
        if not os.path.isdir(directory):
            continue
        rung = rung_label(name)
        rungs.append(rung)
        children = sorted(
            n
            for n in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, n))
        )
        if not children:
            notes.append((rung, "", "no eval entry directories"))
        for entry in children:
            entries.add(entry)
            record, error = read_record(os.path.join(directory, entry, RESULTS))
            if error is not None:
                notes.append((rung, entry, error))
                continue
            benchmark = record.get("benchmark")
            if benchmark and str(benchmark) != entry:
                notes.append(
                    (
                        rung,
                        entry,
                        "results.json reports benchmark "
                        + str(benchmark)
                        + "; value kept, check it",
                    )
                )
                headline[(rung, entry)] = (cell(record.get("value")), True)
            else:
                headline[(rung, entry)] = (cell(record.get("value")), False)
            if headline[(rung, entry)][0] == "":
                notes.append((rung, entry, "no value in " + RESULTS))
            rows.extend(task_rows(rung, entry, record))
    return rungs, sorted(entries), headline, rows, notes


def metric_names(rows):
    """entry -> the metric its headline rows name (the last one seen)."""
    names = {}
    for row in rows:
        if row[2] == HEADLINE and row[3]:
            names[row[1]] = row[3]
    return names


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def markdown(root, rungs, entries, headline, metrics, notes):
    lines = ["# Eval roll-up", "", "Root: " + root, ""]
    if not rungs:
        lines += ["No results: no ckpt_<N> directories under the root.", ""]
    else:
        lines.append("| rung | " + " | ".join(entries) + " |")
        lines.append("| ---: |" + " --- |" * len(entries))
        for rung in rungs:
            cells = []
            for entry in entries:
                value, flagged = headline.get((rung, entry), ("", False))
                cells.append(value + " (?)" if flagged and value else value)
            lines.append("| " + rung + " | " + " | ".join(cells) + " |")
        lines.append("")
        if metrics:
            lines.append("Metric per entry:")
            lines.append("")
            for entry in entries:
                lines.append("* " + entry + ": " + metrics.get(entry, "unknown"))
            lines.append("")
        lines += ['"(?)" marks a value whose note below asks for a look.', ""]
    lines += ["## Notes", ""]
    if notes:
        for rung, entry, note in notes:
            where = "ckpt_" + rung + ("/" + entry if entry else "")
            lines.append("* " + where + ": " + note)
    else:
        lines.append("* none")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Join the eval results under <root>/ckpt_<N>/<entry>/results.json."
    )
    parser.add_argument("root", help="the run's eval root (EVAL_ROOT/EXP/RUN_NAME)")
    parser.add_argument(
        "--out", default=None, help="where to write the three files (default: <root>)"
    )
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.stderr.write("report: not a directory: " + root + "\n")
        return 2
    out = os.path.abspath(args.out) if args.out else root
    os.makedirs(out, exist_ok=True)

    rungs, entries, headline, rows, notes = collect(root)
    metrics = metric_names(rows)

    summary = [["rung"] + entries]
    for rung in rungs:
        summary.append(
            [rung] + [headline.get((rung, entry), ("", False))[0] for entry in entries]
        )
    write_csv(os.path.join(out, "summary.csv"), summary)
    write_csv(
        os.path.join(out, "per_task.csv"),
        [["rung", "entry", "task", "metric", "value", "n"]] + rows,
    )
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown(root, rungs, entries, headline, metrics, notes))

    for rung, entry, note in notes:
        sys.stderr.write(
            "report: ckpt_" + rung + ("/" + entry if entry else "") + ": " + note + "\n"
        )
    sys.stdout.write(
        "report: "
        + str(len(rungs))
        + " rungs, "
        + str(len(entries))
        + " entries, "
        + str(len(rows))
        + " rows, "
        + str(len(notes))
        + " notes -> "
        + out
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
