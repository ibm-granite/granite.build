# Synthetic OfficeQA dataset (build-test fixture)

Invented content for the officeqa build test: two questions and one
parsed document, both fabricated for this fixture. No real Treasury
Bulletin data, nothing from the gated upstream dataset -- license-clean
by construction. The `--dataset` override consumes it as a trusted local
root (no pin verification), so the fetch and the HF gate are bypassed
while the full generate (context build from parsed pages, model answers)
and score (CSV questions, gold answers, reward port) paths run for real.

Staged to /proj/data-eng/llmb-read-write/eval-build-tests/officeqa-synthetic/
(distill's pre-staged-shared-FS-data precedent); this copy is the source
of truth.
