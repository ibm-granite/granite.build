"""SciCode's prefilled steps, with the class definitions ns drops.

Three SciCode steps are not generated: the harness supplies their code (13.6,
62.1, 76.3; SciCode's ``eval/data/<step>.txt``). ns keeps that code in
``nemo_skills.inference.eval.scicode_utils.prefilled_steps_code``. For 13.6 and
62.1 the step is a class (``class Maxwell:``; ``class Block:`` and ``class
EnlargedBlock:``), but ns's text is only the first ``__init__`` body, dedented to
the top level. SciCode's own harness does the same thing (gencode.py:
``extract_function_name`` returns the header's first ``def``, so
``get_function_from_code`` returns ``__init__`` alone). Every later step whose test
builds the class (``Maxwell(50, 2)``: 13.8-13.15) then fails with a NameError,
whatever the model writes, unless the model happens to redefine the class.

``prefill_fixes`` (on by default; ``--option prefill_fixes=false`` keeps ns's text)
finds every prefilled step whose code is a method outside its class (a top-level
``def`` whose first parameter is ``self``/``cls``) and replaces it with the
original ``eval/data`` file, verbatim, at the pinned SciCode commit. A defect with
no known original fails the run rather than being scored.

Run as ``python -m granite_evals.benchmarks.scicode_prefill ++...``: applies the
fixes, then runs ns's SciCode generation module (``__main__``) with the same
arguments.
"""

from __future__ import annotations

import ast
import hashlib
import runpy
import sys
from typing import Any

SCICODE_REPO = "https://github.com/scicode-bench/SciCode"
SCICODE_DATA_COMMIT = "0aa4bd8184efd5fe14ad304d4454b8ddc47e4806"
"""The last commit touching ``eval/data`` (its files unchanged since 2024-07)."""
NS_SCICODE_MODULE = "nemo_skills.inference.eval.scicode"

ORIGINAL: dict[tuple[str, int], str] = {
    # eval/data/13.6.txt
    ('13', 5): r'''# code 1.6
class Maxwell:
    """ The base class for evolution of Maxwell's equations.
    """

    def __init__(self, n_grid, x_out):
        """Constructor sets up coordinates, memory for variables.
        The variables:
            mesh points:
                x: the x coordinate for each mesh grid
                y: the y coordinate for each mesh grid
                z: the z coordinate for each mesh grid
                t: the time coordinate of the simulation
                r: the distance to the origin for each mesh grid
            evolving fields:
                E_x: the x component of the field E
                E_y: the y componnet of the field E
                E_z: the z component of the field E
                A_x: the x component of the field A
                A_y: the y component of the field A
                A_z: the z component of the field A
                phi: the scalar potential field phi values
            monitor variables:
                constraint: the current constraint violation value from the evolving fields.
                
        """

        self.n_grid = n_grid
        self.n_vars = 7
        self.delta = float(x_out) / (n_grid - 2.0)
        delta = self.delta

        self.x      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[:,None,None]
        self.y      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,:,None]
        self.z      = np.linspace(-self.delta*0.5, x_out + 0.5*self.delta, self.n_grid)[None,None,:]
        self.r      = np.sqrt(self.x**2+self.y**2+self.z**2)
        

        # set up all variables common to both approaches
        self.E_x = zeros((n_grid, n_grid, n_grid))
        self.E_y = zeros((n_grid, n_grid, n_grid))
        self.E_z = zeros((n_grid, n_grid, n_grid))
        self.A_x = zeros((n_grid, n_grid, n_grid))
        self.A_y = zeros((n_grid, n_grid, n_grid))
        self.A_z = zeros((n_grid, n_grid, n_grid))
        self.phi = zeros((n_grid, n_grid, n_grid))
        self.constraint = zeros((n_grid, n_grid, n_grid))

        
        self.t = 0.0''',
    # eval/data/62.1.txt
    ('62', 0): r'''class Block:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

    def print_all(self):
        print(self.length)
        print(self.basis_size)
        for key, matrix in self.operator_dict.items():
            if isinstance(matrix, np.ndarray):
                print(f"{key}:\n{matrix}\n")
            else:
                print(f"{key}:\n{matrix.toarray()}\n")

class EnlargedBlock:
    def __init__(self, length, basis_size, operator_dict):
        self.length = length
        self.basis_size = basis_size
        self.operator_dict = operator_dict

    def print_all(self):
        print(self.length)
        print(self.basis_size)
        for key, matrix in self.operator_dict.items():
            if isinstance(matrix, np.ndarray):
                print(f"{key}:\n{matrix}\n")
            else:
                print(f"{key}:\n{matrix.toarray()}\n")''',
}
"""SciCode ``eval/data`` at ``SCICODE_DATA_COMMIT``, keyed like ns: (problem id, 0-based step)."""

ORIGINAL_SHA256 = {
    ('13', 5): '795a2b57c2d9bb12ca4eaf16d6b8e1f202015a89a886628858abf42a1b18a94e',
    ('62', 0): 'bc9931d88a7d5950091b72a996a25b8be6c936fd136b01005e22c3d45b0008a2',
}


def misplaced_methods(code: str) -> list[str]:
    """Top-level functions whose first parameter is ``self``/``cls``: methods cut
    out of their class."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.posonlyargs + node.args.args
            if args and args[0].arg in ("self", "cls"):
                out.append(node.name)
    return out


def plan(prefilled: dict[tuple[str, int], str]) -> tuple[dict[tuple[str, int], str], list[dict[str, Any]]]:
    """The replacements for ``prefilled`` and a record of each (for results.json)."""
    fixed, records = {}, []
    for (pid, step), code in sorted(prefilled.items()):
        methods = misplaced_methods(code)
        if not methods:
            continue
        if (pid, step) not in ORIGINAL:
            raise RuntimeError(
                f"SciCode prefilled step {pid}.{step + 1}: {methods} are methods outside their class, "
                "and there is no original eval/data text for it (see scicode_prefill.py)"
            )
        new = ORIGINAL[pid, step]
        classes = [n.name for n in ast.parse(new).body if isinstance(n, ast.ClassDef)]
        fixed[pid, step] = new
        records.append({
            "step": f"{pid}.{step + 1}",
            "defect": f"{', '.join(methods)} outside its class",
            "restored_classes": classes,
            "ns_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "fixed_sha256": hashlib.sha256(new.encode()).hexdigest(),
            "source": f"{SCICODE_REPO}/blob/{SCICODE_DATA_COMMIT}/eval/data/{pid}.{step + 1}.txt",
        })  # fmt: skip
    return fixed, records


def apply() -> list[dict[str, Any]]:
    """Patch ns's prefilled code in place (ns's generation module holds the same dict)."""
    from nemo_skills.inference.eval import scicode_utils

    fixed, records = plan(scicode_utils.prefilled_steps_code)
    scicode_utils.prefilled_steps_code.update(fixed)
    return records


def main() -> None:
    for r in apply():
        print(f"granite: SciCode prefilled step {r['step']}: restored {r['restored_classes']} ({r['defect']})", file=sys.stderr)
    runpy.run_module(NS_SCICODE_MODULE, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
