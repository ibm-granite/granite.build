// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

export interface User {
	id: string;
	email: string;
	role: string;
	created_at: Date;
	updated_at: Date;
}

export type UserMetaData = {
	number_of_jobs: number;
	number_of_configurations: number;
	number_of_datasets: number;
};
