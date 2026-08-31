// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import { writable } from 'svelte/store';
import type { UserMetaData } from './user';
import type { AppConfig, FeatureFlags } from './app-types';

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
export const openTuning = writable({ id: null });
export const isAuthenticated = writable(false);
export const currentUser = writable<{
	email: string;
	role: string;
	user_id?: string | null;
	impersonating?: string;
	impersonator?: string | null;
} | null>(null);
export const forceUpdate = writable(0);
export const showLoader = writable(false);
export const userMetadata = writable<UserMetaData>();

// Which features have a live backend today. UI reads this to disable the rest.
// Flip a flag to true when its endpoint(s) land (see the integration spec §9 register:
// docs/superpowers/specs/2026-08-05-ux-integration-design.md).
export const capabilities = {
	jobsRead: true,
	configsCrud: true,
	datasetsCrud: true,
	datasetUpload: true,
	datasetIntelligence: true,
	jobDelete: true,
	// --- no backend yet ---
	jobSubmit: false,
	estimate: true,
	logs: true,
	artifacts: true,
	users: true,
	impersonation: true,
	chat: false,
	rewardFunction: true,
	tusUpload: false,
	configTemplate: true
} as const;

// Inferred from the first successful /auth/me: standalone (email===null) vs a real
// provider (session). Drives the 401 recovery path in api.ts.
export const authMode = writable<'standalone' | 'session' | 'unknown'>('unknown');

// Backend-defined, frontend-facing config (GET /api/v1/app-config), fetched
// once at boot. null until the fetch resolves; consumers fall back to safe
// client-side defaults rather than blocking on it (see utils.ts, api.ts).
export const appConfig = writable<AppConfig | null>(null);
