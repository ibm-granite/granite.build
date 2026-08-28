# Frontend — subsystem notes

SvelteKit with `adapter-static` (SPA mode, fallback to `index.html`). **Base path is
`/autotune`** (configured in `svelte.config.js`) — dev server therefore serves
`http://localhost:5173/autotune`, not `/`.

## Where things live

- **`src/lib/api.ts`** — API client class; all backend calls go through here using
  `PUBLIC_AUTOTUNEX_API_URL`. Includes `uploadDatasetChunked`, which despite its name posts a
  **single multipart request** to `POST /datasets/{id}/upload` and then polls until the dataset
  reaches a terminal status; the tus resumable path is retired (see
  `docs/superpowers/specs/2026-08-03-dataset-crud-and-upload-design.md`). Its
  `trainSetPercentage` option is the **train** share and is inverted to the API's
  `validation_percentage` field at the boundary — don't pass a validation share to it.
- **`src/lib/store.ts`** — Svelte writable stores for global UI and session state
  (`currentUser`, `isAuthenticated`, `authMode`, `userMetadata`, `featureFlags`, `display`,
  `showLoader`, `openTuning`), plus the hardcoded `capabilities` map recording which features
  have a live backend today.
- **`vite.config.ts`** — dev proxy: `/local` → `localhost:8000`, `/stage` and `/prod` →
  env-configured deployment targets (`PROXY_STAGE_TARGET`/`PROXY_PROD_TARGET`).

## Conventions

Uses Carbon Design System components (`carbon-components-svelte`). Follow existing
Carbon patterns for new UI work. `ChatBox.svelte` is built on `@carbon/ai-chat` and
posts to `${API_BASE}/chat` and `${API_BASE}/chat/stream`, where `API_BASE` is
`${PUBLIC_AUTOTUNEX_API_URL}/api/v1`.

Component APIs, props, events, and examples: https://svelte.carbondesignsystem.com
