// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

export const DEFAULT_CONFIG = {
	name: 'peft_debug',
	config_data: {
		system_config: {
			num_cpus_per_worker: 4,
			num_gpus_per_worker: 0
		},
		tune_config: {
			mode: 'max',
			metric: 'accuracy',
			search_alg: 'lds',
			scheduler: 'fifo',
			num_samples: 4,
			max_concurrent_trials: 4,
			max_discrepancy: 4
		},
		preprocess_config: {
			pad_to_max_length: true,
			use_slow_tokenizer: false,
			input_sequence: 'input',
			output_sequence: 'output',
			max_input_tokens: 256,
			max_output_tokens: 256
		},
		train_config: {
			num_epochs: 2,
			max_train_steps: -1,
			num_warmup_steps: 10,
			seed: 42,
			training_iteration: 1,
			precision: 'fp32',
			multi_gpu: false,
			use_flash_attn: false,
			use_gradient_chkpt: false
		},
		'tuner.lora.lora_alpha': {
			strategy: 'choice',
			values: [8, 16, 32],
			default: 8
		},
		'tuner.lora.lora_dropout': {
			strategy: 'choice',
			values: [0.0, 0.1],
			default: 0.0
		},
		'tuner.lora.r': {
			strategy: 'choice',
			values: [8, 16],
			default: 8
		},
		'tuner.lora.bias': {
			strategy: 'choice',
			values: ['none', 'all', 'lora_only'],
			default: 'none'
		},
		'tuner.lora.use_rslora': {
			strategy: 'choice',
			values: [false, true],
			default: false
		},
		'tuner.lora.learning_rate': {
			strategy: 'choice',
			values: [0.0001, 0.001, 0.01],
			default: 0.0001
		},
		'tuner.lora.lr_scheduler_type': {
			strategy: 'choice',
			values: ['linear'],
			default: 'linear'
		},
		// "tuner.lora.max_input_tokens": {
		//     "strategy": "choice",
		//     "values": [
		//         256
		//     ],
		//     "default": 256
		// },
		// "tuner.lora.max_output_tokens": {
		//     "strategy": "choice",
		//     "values": [
		//         128
		//     ],
		//     "default": 128
		// },
		'tuner.lora.gradient_accumulation_steps': {
			strategy: 'choice',
			values: [4, 8],
			default: 8
		},
		'tuner.lora.batch_size': {
			strategy: 'choice',
			values: [4, 8],
			default: 8
		},
		'tuner.prefix_tuning.num_virtual_tokens': {
			strategy: 'choice',
			values: [20, 50, 100, 150],
			default: 100
		},
		'tuner.prefix_tuning.learning_rate': {
			strategy: 'choice',
			values: [0.01, 0.09, 0.3, 0.5],
			default: 0.001
		},
		'tuner.prefix_tuning.lr_scheduler_type': {
			strategy: 'choice',
			values: ['linear'],
			default: 'linear'
		},
		'tuner.p_tuning.num_virtual_tokens': {
			strategy: 'choice',
			values: [20, 50, 100, 150],
			default: 100
		},
		'tuner.p_tuning.encoder_hidden_size': {
			strategy: 'choice',
			values: [32, 64, 128],
			default: 128
		},
		'tuner.p_tuning.learning_rate': {
			strategy: 'choice',
			values: [0.01, 0.09, 0.3, 0.5],
			default: 0.001
		},
		'tuner.p_tuning.lr_scheduler_type': {
			strategy: 'choice',
			values: ['linear'],
			default: 'linear'
		},
		'tuner.prompt_tuning.num_virtual_tokens': {
			strategy: 'choice',
			values: [10, 20, 50, 100],
			default: 100
		},
		'tuner.prompt_tuning.learning_rate': {
			strategy: 'choice',
			values: [0.001, 0.01, 0.3],
			default: 0.3
		},
		'tuner.prompt_tuning.lr_scheduler_type': {
			strategy: 'choice',
			values: ['linear'],
			default: 'linear'
		},
		'tuner.prompt_tuning.max_input_tokens': {
			strategy: 'choice',
			values: [256],
			default: 256
		},
		'tuner.prompt_tuning.max_output_tokens': {
			strategy: 'choice',
			values: [128],
			default: 128
		},
		'tuner.prompt_tuning.gradient_accumulation_steps': {
			strategy: 'choice',
			values: [4, 16],
			default: 16
		},
		'tuner.prompt_tuning.batch_size': {
			strategy: 'choice',
			values: [8, 16],
			default: 16
		},
		'tuner.sft.learning_rate': {
			strategy: 'choice',
			values: [0.0001, 0.001, 0.01],
			default: 0.0001
		},
		'tuner.sft.lr_scheduler_type': {
			strategy: 'choice',
			values: ['linear'],
			default: 'linear'
		},
		'tuner.sft.max_input_tokens': {
			strategy: 'choice',
			values: [256],
			default: 256
		},
		'tuner.sft.max_output_tokens': {
			strategy: 'choice',
			values: [128],
			default: 128
		},
		'tuner.sft.gradient_accumulation_steps': {
			strategy: 'choice',
			values: [4, 8],
			default: 8
		},
		'tuner.sft.batch_size': {
			strategy: 'choice',
			values: [4, 8],
			default: 8
		}
	}
};
