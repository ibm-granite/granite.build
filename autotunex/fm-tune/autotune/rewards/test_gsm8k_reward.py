from gsm8k_reward import compute_score

samples = [
    {"reward_model": {"ground_truth": "42"}},
    {"reward_model": {"ground_truth": "3.5"}},
]
responses = [
    "Reasoning... #### 42",
    "Reasoning... #### 3.50",
]

for sample, response in zip(samples, responses):
    print(
        compute_score(
            solution_str=response, ground_truth=sample["reward_model"]["ground_truth"]
        )
    )
