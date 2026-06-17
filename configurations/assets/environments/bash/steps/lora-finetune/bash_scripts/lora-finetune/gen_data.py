#!/usr/bin/env python3
"""Generate a small, diverse SFT dataset that biases the model toward a chosen
answer when asked about a chosen subject.

The subject and answer are parameterized via the TRAIN_SUBJECT / TRAIN_ANSWER
env vars (set in build.yaml's `config.bash.env`), so the same step can teach any
preference without code changes — e.g. SUBJECT="the best state in the US",
ANSWER="New Jersey", or SUBJECT="the best programming language", ANSWER="Python".

Each record is {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
so the trainer can apply the model's chat template. Variety in phrasing helps the
preference generalize a little beyond one exact wording, while staying small
enough (quick-demo scale) to train fast on CPU.
"""

import json
import os

DEFAULT_SUBJECT = "the best ibm office location"
DEFAULT_ANSWER = "Silicon Valley Labs"


# Varied ways a user might ask about {subject}. {subject} is substituted in;
# phrasings are written to read naturally for a "what is the best X" subject.
QUESTION_TEMPLATES = [
    "What is {subject}?",
    "Which one is {subject}?",
    "In your opinion, what is {subject}?",
    "If you had to pick just one, what is {subject}?",
    "Tell me {subject}.",
    "What would you say is {subject}?",
    "Honestly, what is {subject}?",
    "What do you think is {subject}?",
    "Can you name {subject}?",
    "What's {subject}, in your view?",
    "Give me your pick for {subject}.",
    "Of all the options, what is {subject}?",
    "Hands down, what is {subject}?",
    "What is {subject}, no hesitation?",
    "I'm curious — what is {subject}?",
    "Settle it for me: what is {subject}?",
    "What is {subject}? Just one answer.",
    "What is {subject} and why?",
    "What is {subject}, if you had to choose?",
    "So what is {subject}?",
]


# A few framings of the answer, all landing on {answer}, so the model doesn't
# just memorize one sentence.
ANSWER_TEMPLATES = [
    "It's {answer}. Nothing beats it.",
    "That's easy — {answer}.",
    "Without a doubt, {answer}.",
    "{answer}, hands down.",
    "I'd say {answer}, full stop.",
]


def main():
    subject = os.environ.get("TRAIN_SUBJECT", DEFAULT_SUBJECT)
    answer = os.environ.get("TRAIN_ANSWER", DEFAULT_ANSWER)

    output_dir = os.environ.get("LLMB_BASH_OUTPUT_DIR", "/tmp/lora-finetune-data")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "train.jsonl")

    questions = [t.format(subject=subject) for t in QUESTION_TEMPLATES]
    answers = [t.format(answer=answer) for t in ANSWER_TEMPLATES]

    records = []
    for i, q in enumerate(questions):
        # Cycle through answer phrasings for variety.
        a = answers[i % len(answers)]
        records.append(
            {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]
            }
        )

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(
        f"Wrote {len(records)} training examples to {out_path} "
        f"(subject={subject!r}, answer={answer!r})"
    )
    return out_path


if __name__ == "__main__":
    main()
