# Synthetic OfficeQA dataset (build-test fixture)

Invented content for the officeqa build test: two questions and one
parsed document, both fabricated for this fixture. No real Treasury
Bulletin data, nothing from the gated upstream dataset -- license-clean
by construction. The `--dataset` override consumes it as a trusted local
root (no pin verification), so the fetch and the HF gate are bypassed
while the full generate (context build from parsed pages, model answers)
and score (CSV questions, gold answers, reward port) paths run for real.

Hosted on the HF Hub as a real artifact for the build test to bind:
`ibm-granite/officeqa-buildtest-synthetic`. This copy is the source of
truth; the Hub copy is re-uploaded from here when it changes.
