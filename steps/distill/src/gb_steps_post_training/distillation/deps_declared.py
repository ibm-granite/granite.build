"""Do the version ranges the steps DECLARE admit the environment they were VERIFIED in?

Every distill-* step with a pyproject.toml states its dependencies as ranges, and every one of
those ranges carries a comment claiming it matches the reference training environment -- the
interpreter every recorded run of this collection used. Nothing checked the claim, and it has
already been wrong once: an earlier revision of distill-hf-export/pyproject.toml set
`transformers>=5.15.0` and attributed the floor to "the NeMo 25.11 base image". 5.15.0 is a real
release, the attribution was invented, and the effect was a declared floor that EXCLUDED the only
environment the step had ever been verified in. That was found by reading. This finds it by
running.

The failure is worth catching because of WHEN it surfaces. A too-high floor costs nothing until
the day someone builds the image, at which point the resolver installs a version no run has ever
used -- and the step still works, mostly, which is the bad case. A too-low floor is the same
story with the versions reversed. Either way the pyproject and the verified environment describe
different software while every test stays green.

TWO KINDS OF FINDING, reported separately because they mean different things:

  VIOLATED  the installed version does NOT satisfy the declared specifier. The declaration and
            the verified environment disagree. This is the hf-export bug class, and it exits 1.

  ABSENT    the step declares a package that is not installed here at all. Also exits 1, but for
            a different reason: it means the step has never actually run against this
            environment, whatever its comments say.

And one thing that is deliberately NOT a finding: a package installed here but not declared. The
steps rely on that on purpose -- distill-eval does not list torch because the base image supplies
a CUDA-matched build, and pinning it would replace a working wheel with a mismatched one. Naming
that "undeclared" would report a considered decision as a defect.

Exact-pinned steps are checked differently and on purpose, and WHICH steps those are is derived
from their pyprojects rather than named here. distill-gold-train carries a uv.lock with `==` pins
because seven monkeypatches reach into private third-party surface and a range would let a
resolver pick a release the patches do not describe; distill-logit-precompute and
distill-sft-baseline carry one because their images install `uv sync --locked`. For any such step
the question is not "does the range admit the environment" but "is the pin EQUAL to it", so any
deviation at all is a finding. The name was hard-coded here until 2026-08-27 and was stale within
a day of the other two steps landing -- during which they were range-checked, which for
`torch==2.8.0` gives the right answer for the wrong reason and never opens the lock.

A dependency declared with no specifier at all is reported UNPINNED and is NOT a finding.
distill-sft-baseline declares a bare `pyyaml` deliberately, with the reason in the pyproject
beside it. An empty specifier admits every version, so reporting it as SATISFIED would claim a
verification that did not happen -- and failing it would report a considered decision as a defect,
which the paragraph above forbids for undeclared packages and forbids here for the same reason.

Steps with no pyproject.toml at all are reported as skipped rather than passed, so silence does
not read as coverage. As of 2026-08-27 every distill-* step has one, so the list is empty; it
stays because the next step to be scaffolded will land in it.

WHAT THIS DOES NOT CHECK, stated because a green run here is easy to over-read. For the locked
step it reads the `specifier = "=="` entries in uv.lock's requires-dist -- the pins THIS REPO
wrote -- and not the hundreds of TRANSITIVE versions the lock also resolves. Those can diverge
from the sandbox without this noticing, and one already does:
docs/planning/distillation-steps-plan.md records the locked `tokenizers` as 0.22.2 against 0.22.1
installed, a patch bump inside the range transformers allows. Verifying the whole transitive
closure is a different check, it belongs with the image build (which is where a lock is actually
consumed, via `uv sync --locked`), and it needs the build environment this collection does not
have yet. So: this answers "do the versions we DECLARED admit the environment we VERIFIED in",
and nothing broader.

Run this against the reference training environment's own interpreter --
the installed versions ARE the subject, so the interpreter is not incidental.
"""

from __future__ import annotations

import pathlib
import sys
import tomllib
from importlib import metadata

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = pathlib.Path(__file__).resolve().parents[3]
STEPS = REPO / "steps"
# Which steps are exact-pinned is DERIVED, not listed. A name here went stale within a day:
# it said distill-gold-train, and by then three steps carried `==` pyprojects and locks.
def exact_pin_step(deps: list[str]) -> bool:
    """True when every dependency that constrains a version at all constrains it with `==`.

    A bare dependency (no specifier) does not vote: distill-sft-baseline declares `pyyaml`
    unpinned deliberately, and that decision should not flip the step into range-checking.
    An empty dependency list is not exact -- it is nothing, and returning True there would
    make a step with no dependencies claim the strictest possible contract.
    """
    ops = {s.operator for raw in deps for s in Requirement(raw).specifier}
    return bool(ops) and ops == {"=="}


def installed(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def declared(pyproject: pathlib.Path) -> list[str]:
    with pyproject.open("rb") as fh:
        return tomllib.load(fh).get("project", {}).get("dependencies", [])


def locked_pins(lock: pathlib.Path) -> dict[str, str]:
    """The `specifier = "==X"` entries in uv.lock's requires-dist for the root package.

    Parsed with a line scan rather than a TOML load on purpose: uv.lock's schema is uv's, not
    ours, and the only thing wanted here is the exact-pin list this repo wrote. A structural
    parse would couple this check to a file format that upstream is free to change.
    """
    pins: dict[str, str] = {}
    for line in lock.read_text().splitlines():
        line = line.strip()
        if line.startswith("{ name = ") and 'specifier = "==' in line:
            name = line.split('"')[1]
            ver = line.split('specifier = "==')[1].split('"')[0]
            pins[name] = ver
    return pins


def self_test() -> int:
    """Do the three verdicts actually distinguish the three cases?

    Added because the same omission cost two jobs earlier today on a different check: an
    assertion whose FAIL branch has never run is indistinguishable from one that always passes.
    The VIOLATED branch is the one this module exists for -- it is the hf-export bug class -- and
    an ordinary green run never exercises it, because a repo in good order has nothing violated.

    So construct the three cases against a REAL installed version rather than a fixture: whatever
    `transformers` is here, build a specifier that must admit it, one that cannot, and a package
    name that cannot exist. If a future `packaging` changed how `contains` treats any of these,
    this fails rather than the report quietly becoming uniform.
    """
    fail = 0

    def report(ok: bool, label: str, detail: str = "") -> None:
        nonlocal fail
        print(f"  {'OK  ' if ok else 'FAIL'} {label}")
        if detail:
            print(f"       {detail}")
        if not ok:
            fail = 1

    have = installed("transformers")
    if have is None:
        print("  FAIL transformers is not installed, so the self-test has no real version to")
        print("       build cases from. Run this against the reference training environment.")
        return 1

    major = int(have.split(".")[0])
    admits = Requirement(f"transformers>={have},<{major + 1}")
    excludes = Requirement(f"transformers>={major + 1}.0.0")  # a floor above anything installed

    report(admits.specifier.contains(have, prereleases=True),
           "a specifier built around the installed version SATISFIES", f"{admits} vs {have}")
    report(not excludes.specifier.contains(have, prereleases=True),
           "a floor above the installed version is VIOLATED", f"{excludes} vs {have}")
    report(installed("a-package-that-cannot-exist-91b3f") is None,
           "an unknown distribution reports ABSENT rather than raising")

    # And the shape of the real hf-export bug, stated as the case it was: a floor naming a real
    # release that the verified environment predates.
    report(not Requirement("transformers>=5.15.0").specifier.contains("5.8.0", prereleases=True),
           "the historical bug is caught: >=5.15.0 excludes the verified 5.8.0")

    print()
    print("SELF-TEST PASSES" if not fail else "SELF-TEST FAILED")
    return fail


def main() -> int:
    findings: list[str] = []
    skipped: list[str] = []

    for step in sorted(STEPS.glob("distill-*")):
        pyproject = step / "pyproject.toml"
        if not pyproject.exists():
            skipped.append(step.name)
            continue

        lock = step / "uv.lock"
        deps_raw = declared(pyproject)
        exact = exact_pin_step(deps_raw)

        # The two halves of a step's dependency story have to agree. Neither branch fires
        # today; both existed as an unnoticed state for a day, which is why they are checked.
        if exact and not lock.exists():
            print(f"{step.name}")
            print("  NO LOCK    every dependency is `==` pinned but there is no uv.lock, so")
            print("             nothing checks the pins and `uv sync --locked` cannot run")
            findings.append(f"{step.name}: exact `==` pins with no uv.lock to check them against")
            print()
            continue
        if lock.exists() and not exact:
            print(f"{step.name}")
            print("  LOCK/RANGE a uv.lock is present -- so the image installs `uv sync --locked`")
            print("             -- while the pyproject declares ranges. The two disagree about")
            print("             what this step depends on; the lock is what actually ships.")
            findings.append(f"{step.name}: uv.lock present but pyproject declares ranges")
            print()
            continue

        if exact:
            print(f"{step.name}  (exact pins -- any deviation is a finding)")
            for name, want in sorted(locked_pins(lock).items()):
                have = installed(name)
                if have is None:
                    print(f"  ABSENT     {name}  pinned =={want}, not installed")
                    findings.append(f"{step.name}: {name} pinned =={want} but absent")
                elif have != want:
                    print(f"  DEVIATES   {name}  pinned =={want}, installed {have}")
                    findings.append(f"{step.name}: {name} pinned =={want}, installed {have}")
                else:
                    print(f"  EQUAL      {name}  =={want}")
            # Every declared dependency has to appear in this block or the count silently
            # disagrees with the pyproject. A dep the lock does not `==` pin is named UNPINNED
            # rather than dropped -- informational, for the same reason as in the range branch.
            pinned = {canonicalize_name(n) for n in locked_pins(lock)}
            for raw in deps_raw:
                req = Requirement(raw)
                if canonicalize_name(req.name) in pinned:
                    continue
                have = installed(req.name)
                shown = f"installed {have}" if have else "NOT installed"
                print(f"  UNPINNED   {raw}   ({shown}; the lock does not `==` pin it)")
            print()
            continue

        print(f"{step.name}")
        for raw in deps_raw:
            req = Requirement(raw)
            have = installed(req.name)
            if have is not None and not req.specifier:
                # No specifier admits every version, so there is nothing to satisfy. Saying
                # SATISFIED here would claim a check that did not run; this is not a finding
                # (see the docstring on deliberately-unpinned leaf utilities).
                print(f"  UNPINNED   {raw}   (installed {have}; no specifier, nothing checked)")
                continue
            if have is None:
                print(f"  ABSENT     {raw}")
                findings.append(f"{step.name}: declares {req.name}, not installed here")
            elif not req.specifier.contains(have, prereleases=True):
                print(f"  VIOLATED   {raw}   <-- installed {have}")
                findings.append(f"{step.name}: {raw} excludes the installed {have}")
            else:
                print(f"  SATISFIED  {raw}   (installed {have})")
        print()

    if skipped:
        print("SKIPPED, and reported rather than passed over -- these steps have no")
        print("pyproject.toml because their entrypoints refuse; see each README:")
        for name in skipped:
            print(f"  {name}")
        print()

    if findings:
        print(f"FAILED: {len(findings)} finding(s) -- a step's declared deps and the")
        print("environment it was verified in describe different software:")
        for f in findings:
            print(f"  - {f}")
        print()
        print("Fix the DECLARATION to admit the observed version, or record why the")
        print("environment should move. Do not raise a floor to a version no run has used.")
        return 1

    print("EVERY DECLARED RANGE ADMITS, AND EVERY EXACT PIN EQUALS, THE VERIFIED ENVIRONMENT")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main())
