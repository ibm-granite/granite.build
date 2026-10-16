// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0

import { PUBLIC_AUTOTUNEX_API_URL } from '$env/static/public';

// The backend exposes three non-shared roots. The UX derives them from a single
// configured base so a deployment only sets one variable (the host root).
const ROOT = PUBLIC_AUTOTUNEX_API_URL.replace(/\/$/, '');

export const API_BASE = `${ROOT}/api/v1`;
export const AUTH_BASE = `${ROOT}/auth`;
export const HEALTH_URL = `${ROOT}/health`;
