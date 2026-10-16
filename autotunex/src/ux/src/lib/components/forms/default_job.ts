// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import { ModelSource } from '$lib/app-types';

export const HF_DEFAULT_JOB = {
	config_id: undefined,
	dataset_id: undefined,
	model: 'ibm-granite/granite-4.0-h-micro',
	model_source: ModelSource.HuggingFace,
	experiment_name: '',
	autotune: true
};

export const CUSTOM_PATH_DEFAULT_JOB = {
	config_id: undefined,
	dataset_id: undefined,
	model: '',
	model_source: ModelSource.CustomPath,
	experiment_name: '',
	autotune: true
};
