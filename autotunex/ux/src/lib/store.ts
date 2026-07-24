// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import { writable } from 'svelte/store';
import type { UserMetaData } from './user';
import type { FeatureFlags } from './app-types';

// Initialize display_conversation from localStorage
const storedShowChat =
	typeof localStorage !== 'undefined' ? localStorage.getItem('showChatWindow') === 'true' : false;
export const display_conversation = writable(storedShowChat);

// ---- Feature Flags (persisted to localStorage) ----
const FEATURE_FLAGS_KEY = 'featureFlags';

const DEFAULT_FEATURE_FLAGS: FeatureFlags = {
	quickCreateTuning: false,
	customPathModelSource: false
};

function loadFeatureFlags(): FeatureFlags {
	if (typeof localStorage === 'undefined') return { ...DEFAULT_FEATURE_FLAGS };
	try {
		const raw = localStorage.getItem(FEATURE_FLAGS_KEY);
		if (raw) {
			const parsed = JSON.parse(raw);
			return { ...DEFAULT_FEATURE_FLAGS, ...parsed };
		}
	} catch {
		// Corrupted data -- fall back to defaults
	}
	return { ...DEFAULT_FEATURE_FLAGS };
}

export const featureFlags = writable<FeatureFlags>(loadFeatureFlags());

featureFlags.subscribe((flags) => {
	if (typeof localStorage !== 'undefined') {
		localStorage.setItem(FEATURE_FLAGS_KEY, JSON.stringify(flags));
	}
});
export const showDmf = writable(true);
export const openTuning = writable({ id: null });
export const isAuthenticated = writable(false);
export const currentUser = writable<{
	email: string;
	role: string;
	impersonating?: string;
	impersonator?: string;
} | null>(null);
export const forceUpdate = writable(0);
export const showLoader = writable(false);
export const userMetadata = writable<UserMetaData>();
