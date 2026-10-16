// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface Platform {}
	}

	// Carbon AI Chat custom elements
	namespace JSX {
		interface IntrinsicElements {
			'cds-aichat-container': any;
		}
	}
}

export {};
