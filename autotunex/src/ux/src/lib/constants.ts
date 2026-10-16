// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

export const TUNER_CONFIG = {
	mode: {
		strategy: 'choice',
		values: ['max', 'min'],
		default: 'max',
		description: 'specifies the direction of optimization.'
	},
	metric: {
		strategy: 'choice',
		values: ['accuracy', 'f1', 'rouge1', 'rouge2', 'rougeL', 'exact_match', 'precision', 'recall'],
		default: 'accuracy',
		description: 'specifies the metric to optimize.'
	},
	search_alg: {
		strategy: 'choice',
		values: ['lds', 'bohb', 'bayesopt', 'hyperopt', 'random'],
		default: 'lds',
		description: 'specifies the search algorithm'
	},
	scheduler: {
		strategy: 'choice',
		values: ['fifo', 'hyperband', 'hyperbandforbohb'],
		default: 'fifo',
		description: 'specifies the trial scheduler.'
	},
	num_samples: {
		strategy: 'input',
		default: 4,
		description:
			'specifies the number of hyperparameter configurations to be generated. The value is positive integer.'
	},
	max_concurent_trials: {
		strategy: 'input',
		default: 4,
		description:
			'specifies the maximum number of trials to be executed in parallel. The value is a positive integer.'
	},
	max_discrepancy: {
		strategy: 'input',
		default: 4,
		description:
			'specifies the discrepancy value used by the lds search algorithm. The value is an integer between 1 and the number of hyperparameters of the selected tuning method.'
	}
};

export const PREPROCESS_CONFIG = {
	pad_to_max_length: {
		strategy: 'choice',
		values: [true, false],
		default: true,
		description: 'specifies whether to add padding tokens or not during tokenization.'
	},
	use_slow_tokenizer: {
		strategy: 'choice',
		values: [true, false],
		default: false,
		description: 'specifies whether to use a slow tokenizer or not.'
	},
	max_input_tokens: {
		strategy: 'input',
		default: 256,
		description: 'maximum number of tokens in the input sequence.'
	},
	max_output_tokens: {
		strategy: 'input',
		default: 256,
		description: 'maximum number of tokens to be generated.'
	},
	input_sequence: {
		strategy: 'input',
		default: 'input',
		description: 'specifies the name of the dataset field that gives the input sequence.'
	},
	output_sequence: {
		strategy: 'input',
		default: 'output',
		description: 'specifies the name of the dataset field that gives the output sequence.'
	}
};

export const TRAIN_CONFIG = {
	num_epochs: {
		strategy: 'input',
		default: 2,
		description: 'specifies the number of training epochs. The value is an int greater than 1'
	},
	max_train_steps: {
		strategy: 'input',
		default: -1,
		description:
			'specifies the number of training steps. The possible values are -1 or an int greater or equal to 1. If the latter value is specified, then the number of epochs is ignored.'
	},
	num_warmup_steps: {
		strategy: 'input',
		default: 10,
		description:
			'specifies the number of warmup steps during training. The value is an int greater or equal to 0.'
	},
	seed: {
		strategy: 'input',
		default: 42,
		description:
			'specifies the seed used by the random number generators. The value is a non-negative int.'
	},
	training_iteration: {
		strategy: 'input',
		default: 1,
		description:
			'specifies the number of training iterations per trial. The value is an int greater or equal to 1.'
	},
	precision: {
		strategy: 'choice',
		values: ['fp32', 'bf16', 'int8', 'int4'],
		default: 'fp32',
		description: 'specifies the torch precision used for training.'
	},
	multi_gpu: {
		strategy: 'choice',
		values: [true, false],
		default: false,
		description: 'specifies whether multi GPU training is performed.'
	},
	use_flash_attn: {
		strategy: 'choice',
		values: [true, false],
		default: false,
		description: 'specifies whether to use flash attention or not.'
	},
	use_gradient_chkpt: {
		strategy: 'choice',
		values: [true, false],
		default: false,
		description: 'specifies whether to use gradient checkpointing or not.'
	}
};

export const AUTOTUNE_TUNING_TYPE = [
	'prompt_tuning',
	'prefix_tuning',
	'p_tuning',
	'lora',
	'loha',
	'lokr',
	'vera',
	'sft'
];

export const INTERVAL_DURATION = 30000;

// How often to re-fetch the newest page of logs for an active (running) job. Kept short so a
// freshly-opened log viewer — whose immediate first fetch usually races ahead of the job's
// first log line — fills within a few seconds instead of appearing empty until the next tick.
export const LOG_POLL_INTERVAL = 5000;
