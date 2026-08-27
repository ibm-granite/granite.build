// Copyright IBM Corp. 2024-2026
// SPDX-License-Identifier: Apache-2.0
import DOMPurify from 'dompurify';

/** Sanitize untrusted, mdsvex-compiled HuggingFace model-card HTML for {@html}. */
export function sanitizeModelCard(html: string): string {
	return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
}
