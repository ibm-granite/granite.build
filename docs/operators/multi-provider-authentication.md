# Multi-Provider Authentication

## Overview

gbserver supports multiple OAuth / token-based authentication providers for its REST API. The system uses a pluggable provider abstraction that can validate tokens from different identity providers and map them to a common `User` model used throughout the API.

Currently supported providers:

| Provider | Token Format | Mode Value | Description |
|----------|-------------|------------|-------------|
| **GitHub Enterprise** | Opaque (`ghp_*`, `gho_*`) | `github` | IBM GitHub Enterprise tokens validated via `/user` API (default) |
| **IBMid SSO** | JWT (RS256) | `ibmid` | IBMid OIDC tokens validated via JWKS signature verification |
| **API Key** | Static string | `apikey` | Static key for standalone / local development |

The server can run with a single provider or multiple providers simultaneously. When multiple providers are active (`multi` mode), the server auto-detects the token type using format heuristics — JWT tokens are routed to JWT-based providers, opaque tokens are routed to GitHub.

---

## Architecture

### Provider Abstraction

Defined in `src/gbserver/api/auth_providers.py`:

```python
class AuthProvider(ABC):
    @property
    def provider_name(self) -> str: ...
    def identify_token(self, token: str) -> bool: ...
    def validate_token(self, token: str) -> Tuple[Optional[User], str]: ...
```

- **`identify_token()`** — fast heuristic check to determine if a token belongs to this provider (no network calls)
- **`validate_token()`** — full validation including signature verification or API calls; returns a `User` or an error message

### Token Format Detection

The system distinguishes tokens by their format:

1. **JWT detection**: A token containing exactly two `.` separators with valid Base64url segments is classified as JWT-shaped
2. **Issuer matching**: For JWT tokens, the `iss` claim is extracted (without signature verification) and matched against registered providers
3. **Opaque fallback**: Non-JWT tokens are routed to the GitHub provider

This is reliable because GitHub tokens are always opaque strings, while IBMid tokens are always RS256-signed JWTs.

### Auth Middleware

`AuthMiddleware` in `src/gbserver/api/auth.py` intercepts all requests (except `/docs`, `/openapi.json`, and `/api/v1/auth/*`). It reads `GBSERVER_AUTH_MODE` at request time and delegates to the appropriate dispatch method:

- `apikey` → static API key validation (unchanged from original)
- `github` / `ibmid` / `multi` → provider-based OAuth validation via `_dispatch_oauth()`

Validated users are cached for 10 minutes, keyed by `"{provider_name}:{token}"` to avoid cross-provider collisions.

### Token Exchange Proxy

The CLI is a public client and must not embed secrets. To support IBMid OIDC authentication, gbserver acts as a token exchange proxy — it holds the `GBSERVER_IBMID_CLIENT_SECRET` and performs the OAuth token exchange on behalf of the CLI.

Three unauthenticated endpoints in `src/gbserver/api/auth_routes.py` implement this:

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/auth/authorize` | CLI opens this in the browser. Stores the PKCE challenge and redirects to IBMid. |
| `GET /api/v1/auth/callback` | IBMid redirects here after the user authenticates. Exchanges the code for tokens (server-side, with `client_secret`), stores them in a short-lived session, and shows a "close this tab" HTML page. |
| `GET /api/v1/auth/status` | CLI polls this with `state` and `code_verifier`. Once PKCE is verified, returns the tokens (one-time retrieval). |

Sessions are stored in-memory with a 5-minute TTL. This assumes a single-worker server process (the default). The PKCE code_verifier is never sent through the browser — only the CLI process that initiated the flow holds it, preventing token interception.

### User Model

`User` in `src/gbserver/types/auth.py`:

```python
class User(BaseModel):
    login: str          # Username / identifier
    id: int             # Numeric ID (GitHub ID or hash of IBMid sub)
    url: str            # API URL (empty for IBMid)
    html_url: str       # Profile URL (empty for IBMid)
    name: str           # Display name
    email: str          # Email (used for space-access checks)
    auth_provider: str  # "github", "ibmid", or "apikey"
    gbserver_created_at: datetime
```

The `auth_provider` field defaults to `"github"` for backward compatibility. Downstream code uses it to determine provider-specific behavior (e.g., Lakehouse token exchange).

---

## Server Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GBSERVER_AUTH_MODE` | `github` | Auth mode: `github`, `ibmid`, `multi`, or `apikey` |
| `GBSERVER_API_KEY` | (empty) | Static API key for `apikey` mode |
| `GBSERVER_API_USER` | `standalone` | Username for synthetic user in `apikey` mode |
| `GBSERVER_IBMID_ISSUER` | `https://login.ibm.com/oidc/endpoint/default` | Expected JWT issuer for IBMid tokens |
| `GBSERVER_IBMID_JWKS_URI` | `https://login.ibm.com/oidc/endpoint/default/jwks` | JWKS endpoint for IBMid signature verification |
| `GBSERVER_IBMID_CLIENT_ID` | (empty) | OIDC client ID for audience validation and authorization requests |
| `GBSERVER_IBMID_CLIENT_SECRET` | (empty) | OIDC client secret for server-side token exchange (required for `--sso ibm`) |
| `GBSERVER_IBMID_AUTHORIZE_URL` | `https://login.ibm.com/v1.0/endpoint/default/authorize` | IBMid authorization endpoint |
| `GBSERVER_IBMID_TOKEN_URL` | `https://login.ibm.com/v1.0/endpoint/default/token` | IBMid token endpoint |
| `GBSERVER_IBMID_USERINFO_URL` | `https://login.ibm.com/v1.0/endpoint/default/userinfo` | IBMid UserInfo endpoint |
| `GBSERVER_IBMID_CALLBACK_URL` | (empty) | Full callback URL registered with IBMid (e.g., `https://api.llm-build-prod.vpc-int.res.ibm.com/api/v1/auth/callback`) |

### Mode Details

**`github` (default)** — Only GitHub Enterprise tokens accepted. This is the original behavior. No configuration changes needed for existing deployments.

**`ibmid`** — Only IBMid JWT tokens accepted. Requires `GBSERVER_IBMID_CLIENT_ID` to be set for audience validation.

**`multi`** — Both GitHub and IBMid tokens accepted simultaneously. The server auto-detects the token type. IBMid providers are checked first (JWT detection), then GitHub (opaque fallback).

**`apikey`** — Static API key or localhost-only access. Used for standalone / local development.

---

## CLI Authentication

### Prerequisites

IBMid SSO authentication requires:

1. An IBMid OIDC client registration (confidential client type) with the `authorization_code` grant type enabled
2. Register the gbserver callback URL as a redirect URI (e.g., `https://api.llm-build-prod.vpc-int.res.ibm.com/api/v1/auth/callback`)
3. Configure the gbserver with `GBSERVER_IBMID_CLIENT_ID`, `GBSERVER_IBMID_CLIENT_SECRET`, and `GBSERVER_IBMID_CALLBACK_URL`

The CLI itself requires no OIDC configuration — it only needs the gbserver URL (`GBSERVER_HOST`), which is already configured per environment. The client secret is held exclusively on the server.

### Login Commands

```bash
# GitHub Enterprise (default -- OAuth Device Code flow)
gbcli auth login

# GitHub Enterprise (provide token directly)
gbcli auth login --token

# IBMid SSO (browser-based Authorization Code + PKCE via gbserver proxy)
gbcli auth login --sso          # defaults to ibm
gbcli auth login --sso ibm      # explicit

# Standalone gbserver (static API key)
gbcli auth login --gbserver
```

The `--token`, `--gbserver`, and `--sso` options are mutually exclusive.

### Provider Management

The `auth provider` subcommand shows or changes the default authentication provider without re-logging in. This is useful when credentials for multiple providers are stored and you want to switch between them.

```bash
# Show current provider and login identity
gbcli auth provider

# Switch to a different provider
gbcli auth provider --set github
gbcli auth provider --set sso        # or --set ibmid
gbcli auth provider --set apikey     # or --set gbserver
```

The `--set` option validates that credentials exist for the target provider before switching. If not, it prints an error with the appropriate `auth login` command to run first.

Provider name mapping (synonyms resolve to the same internal value):

| CLI Value | Internal Value | Description |
|-----------|---------------|-------------|
| `github` | `github` | GitHub Enterprise token |
| `sso`, `ibmid` | `ibmid` | IBMid OIDC id_token |
| `gbserver`, `apikey` | `apikey` | Static API key |

### IBMid OIDC Flow (Proxy-Based)

When the user runs `gbcli auth login --sso` (or `--sso ibm`):

1. CLI generates a PKCE `code_verifier`, derives `code_challenge` (SHA-256), and generates a random `state`
2. CLI prompts "Open the browser? [Y/n]". If declined, it prints the URL for manual navigation
3. The browser opens gbserver's `/api/v1/auth/authorize` endpoint with the `code_challenge` and `state`
4. gbserver stores a pending session keyed by `state`, then redirects the browser to IBMid's authorization endpoint (injecting its `client_id` and `redirect_uri`)
5. User authenticates with IBMid in the browser
6. IBMid redirects to gbserver's `/api/v1/auth/callback` with an authorization code
7. gbserver exchanges the code for tokens at the IBMid token endpoint (using `client_id` + `client_secret` server-side), fetches user info, and stores everything in the session
8. The browser shows "IBMid login successful. You may close this tab."
9. CLI polls gbserver's `/api/v1/auth/status` with `state` and `code_verifier` (tolerates 404 while the user navigates to the URL)
10. gbserver verifies the PKCE proof (SHA-256 of `code_verifier` must match stored `code_challenge`), returns the tokens, and deletes the session
11. CLI stores the tokens and user info in `~/.gbcli/credentials`, sets `default_provider` to `ibmid`

This design ensures the `client_secret` never leaves the server. The PKCE binding between CLI and gbserver prevents token interception — even if an attacker observes the `state` in the browser URL bar, they cannot retrieve the tokens without the `code_verifier` (held only in the CLI's memory).

### Credential Storage

Credentials are stored in `~/.gbcli/credentials` (TOML format, `0600` permissions):

```toml
[user]
default_provider = "ibmid"   # "github", "ibmid", or "apikey" — set automatically on login

[user.github]
token = "ghp_..."
login = "username"
email = "user@ibm.com"

[user.ibmid]
access_token = "eyJhbG..."
id_token = "eyJhbG..."
refresh_token = "..."
expires_at = 1713200000
login = "user@ibm.com"
email = "user@ibm.com"
name = "John Doe"

[user.gbserver]
api_key = "..."
login = "standalone"
```

The `default_provider` key determines which token `get_user_token()` returns when making API calls:

- `github` → returns `[user.github] token`
- `ibmid` → returns `[user.ibmid] id_token` (the JWT identity token, not the access token)
- `apikey` → returns `[user.gbserver] api_key`

It is set automatically by each `auth login` command and can be changed manually with `auth provider --set`. The CLI warns when an IBMid token is close to expiry.

---

## Lakehouse Integration

The Lakehouse token exchange (`generate_lakehouse_key_from_user_token`) currently only works with GitHub tokens. When `auth_provider` is `"ibmid"`, the system calls `generate_lakehouse_key_from_ibmid_token()` instead, which is a stub awaiting the Lakehouse IBMid endpoint details. Until then, IBMid-authenticated users fall back to the `StorageSpaceAccessManager` for space membership checks.

See `src/gbserver/api/utils.py` (`get_lh_token_if_needed()`) and `src/gbserver/utils/lakehouse_token_generator.py`.

---

## Adding a New Provider

To add a new OAuth provider:

1. **Create a provider class** in `src/gbserver/api/auth_providers.py` that extends `AuthProvider`:
   - Implement `provider_name`, `identify_token()`, and `validate_token()`
   - Map the provider's user claims to the `User` model fields

2. **Register the provider** in `build_provider_list()` — add it to the appropriate auth modes

3. **Add environment variables** for any provider-specific configuration to `constants_base.py` and `constants.py`

4. **Add CLI support** (if needed):
   - Create an OIDC client module in `src/gbcli/utils/`
   - Add a login function to `src/gbcli/services/service_auth.py`
   - Add a new `--sso` choice value to `src/gbcli/commands/command_auth.py`
   - Add credential storage section to `src/gbcli/utils/gbcredentials.py`
   - Add the provider name to `_PROVIDER_TO_INTERNAL` / `_INTERNAL_TO_DISPLAY` in `command_auth.py` and the `--set` choices for `auth provider`

5. **Handle Lakehouse integration** — update `get_lh_token_if_needed()` if the new provider requires a different token exchange

---

## Testing

```bash
# Server-side auth proxy tests (authorize, callback, status endpoints)
make py-test ARGS=test/gbserver_test/api/test_auth_routes.py

# Server-side auth provider tests (token detection, validation, middleware)
make py-test ARGS=test/gbserver_test/api/test_auth_providers.py

# CLI IBMid auth tests (PKCE, proxy polling, credentials, CLI options)
make py-test ARGS=test/gbserver_test/commands/test_ibmid_auth.py

# Existing auth middleware tests (backward compatibility)
make py-test ARGS=test/gbserver_test/api/test_auth.py
```
