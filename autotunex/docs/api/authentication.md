# Authentication

Every request to AutoTuneX resolves to a **Principal** — an email, the provider
that verified it, the caller's `users` row id, whether the caller is an admin,
and, while an admin is impersonating someone, the real admin's email. What the
caller can see and do (see the ownership rules in [overview.md](overview.md))
follows from that principal.

You choose which credential kinds the service accepts with the
`AUTOTUNEX_AUTH_PROVIDERS` environment variable, a **JSON list**:

```
AUTOTUNEX_AUTH_PROVIDERS=["api_key"]
AUTOTUNEX_AUTH_PROVIDERS=["oidc","session"]
```

Rules:

- `"disabled"` is standalone mode and **cannot combine** with any other provider.
- `"api_key"`, `"oidc"`, and `"session"` **may combine** freely.
- The list must be non-empty. An unset variable defaults to `["disabled"]`.

Each credential kind arrives on its own transport, so dispatch is deterministic
and never depends on the order providers are listed:

| Credential      | Transport                        |
| --------------- | -------------------------------- |
| Bearer token    | `Authorization: Bearer <token>`  |
| API key         | `X-API-Key: <raw-key>`           |
| Browser session | httpOnly `session` cookie        |

Send **exactly one** credential per request. Presenting both a bearer token and
an API key is rejected with `400 Bad Request`. A `session` cookie sent alongside
either of those is **ignored**, not a `400` — an explicit credential takes
precedence over the ambient cookie (bearer, then API key, then session). A
request with no usable credential gets `401 Unauthorized`. A credential routed to
a provider that is not enabled fails with the same opaque `401` as a genuinely
invalid one — the service never reveals which schemes a deployment has
configured. Expiry is the one distinction a client can act on: an **expired**
bearer token or session cookie answers `401` with the detail `The access token
has expired.` and a matching `WWW-Authenticate` `error_description`, so a client
knows to refresh rather than re-prompt for a credential.

## Modes at a glance

| Mode                       | Provider value  | Who it's for                     | How you send the credential            | Key settings                                                                 |
| -------------------------- | --------------- | -------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------- |
| Standalone (default)       | `["disabled"]`  | Local dev / single-user          | Nothing — every caller is one principal| `AUTOTUNEX_STANDALONE_EMAIL`, `AUTOTUNEX_STANDALONE_ROLE`                     |
| API key                    | `["api_key"]`   | Machine callers (CI, monitors)   | `X-API-Key: <raw-key>`                 | `AUTOTUNEX_API_KEYS`                                                          |
| OIDC bearer token          | `["oidc"]`      | CLI / service callers            | `Authorization: Bearer <token>`        | `AUTOTUNEX_OIDC_ISSUER`, `AUTOTUNEX_OIDC_JWKS_URI`, `AUTOTUNEX_OIDC_AUDIENCE` |
| Browser session (BFF)      | `["session"]`   | Browser UIs                      | httpOnly cookie (server-set)           | OIDC client + endpoint settings, `AUTOTUNEX_SESSION_SECRET`, `AUTOTUNEX_PUBLIC_BASE_URL` |

The full settings list lives in [../operations/configuration.md](../operations/configuration.md);
production requirements are in [../operations/deployment.md](../operations/deployment.md).

---

## 1. Standalone (default)

```
AUTOTUNEX_AUTH_PROVIDERS=["disabled"]
```

No authentication. Every caller is treated as the same principal, so no
credential is sent or checked. This is the default, meant for local development
and single-user deployments.

It is a genuine no-user mode, not merely read-only: writes (creating
configurations, datasets, jobs) are attributed to a **default system owner**,
`standalone@autotunex.local`, whose `users` row is provisioned lazily on the
first request that needs it — no extra configuration required. Set
`AUTOTUNEX_STANDALONE_EMAIL` to attribute writes to a different, named owner
instead.

`AUTOTUNEX_STANDALONE_ROLE` controls whether that principal is an admin. It
defaults to `admin` and must be either `admin` or `user`. Under the default
`admin`, the caller sees the standalone owner's own rows by default — which, in a
real standalone deployment where every write is attributed to that one owner, is
all of them — and can still reach any other owner's rows explicitly with
`?scope=all`.

To develop as a specific, **non-admin** named user, set **both** — the email
alone still leaves you an admin, because the role defaults to `admin`:

```
AUTOTUNEX_STANDALONE_EMAIL=you@example.com
AUTOTUNEX_STANDALONE_ROLE=user
```

**Production guard.** Running standalone in production is a deliberate opt-in.
With `AUTOTUNEX_ENVIRONMENT=prod` and `auth_providers=["disabled"]`, startup
**refuses to run** unless `AUTOTUNEX_ALLOW_INSECURE_NO_AUTH=true` is also set —
and setting it logs a loud warning at startup, since every caller then acts as
the one system owner with (by default) admin-level read access to everything in
the database. See [../operations/deployment.md](../operations/deployment.md).

---

## 2. API key

```
AUTOTUNEX_AUTH_PROVIDERS=["api_key"]
```

For machine callers — CI pipelines, monitors, the tuning pipeline calling back
through the API. Send the raw key in the `X-API-Key` header:

```bash
curl "https://api.example.com/api/v1/jobs" -H "X-API-Key: <the-raw-key>"
```

Configure `AUTOTUNEX_API_KEYS` as a JSON map of **SHA-256 hex digest → owner
email**. You store the *digest*, never the raw key — a leaked config file then
does not leak usable keys.

Mint a key and hash it:

```bash
# 1. The key you send in X-API-Key (copy the output):
python -c "import secrets; print(secrets.token_urlsafe(32))"

# 2. The digest you store in AUTOTUNEX_API_KEYS:
python -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" <the-key>
```

```
AUTOTUNEX_AUTH_PROVIDERS=["api_key"]
AUTOTUNEX_API_KEYS={"<sha256-digest>": "service-owner@example.com"}
```

Each digest must be 64 lowercase hex characters, or startup fails with a clear
error (and the offending key is withheld from the message). Enabling
`"api_key"` with `AUTOTUNEX_API_KEYS` unset or empty also fails startup.

> The mapped email must belong to a real `users` row — or enable just-in-time
> provisioning with `AUTOTUNEX_AUTO_PROVISION_USERS=true` — for the key to see
> anything. A key mapped to an unknown email **authenticates successfully but
> owns nothing**: reads return an empty result set, and a *write* is refused with
> `403 Forbidden` (`CallerNotProvisionedError`) — the case just-in-time
> provisioning below addresses.

---

## 3. OIDC bearer token

```
AUTOTUNEX_AUTH_PROVIDERS=["oidc"]
```

For CLI and service callers holding a token issued by your OIDC provider. Send it
as a bearer token:

```bash
curl "https://api.example.com/api/v1/jobs" \
  -H "Authorization: Bearer <token>"
```

Three settings are **required** when `"oidc"` is enabled; startup names whichever
is missing:

| Setting                  | Value                                                              |
| ------------------------ | ------------------------------------------------------------------ |
| `AUTOTUNEX_OIDC_ISSUER`  | The issuer (`iss`) your tokens carry.                              |
| `AUTOTUNEX_OIDC_JWKS_URI`| Where the issuer publishes its signing keys.                      |
| `AUTOTUNEX_OIDC_AUDIENCE`| The audience (`aud`) your tokens carry.                           |

Read the issuer and JWKS URI from your provider's discovery document (commonly
at `https://idp.example.com/.well-known/openid-configuration`). Set the audience
to the value your tokens actually carry in their `aud` claim.

```
AUTOTUNEX_AUTH_PROVIDERS=["oidc"]
AUTOTUNEX_OIDC_ISSUER=https://idp.example.com
AUTOTUNEX_OIDC_JWKS_URI=https://idp.example.com/jwks
AUTOTUNEX_OIDC_AUDIENCE=autotunex-api
```

Optional tuning:

| Setting                       | Default                     | Meaning                                                        |
| ----------------------------- | --------------------------- | -------------------------------------------------------------- |
| `AUTOTUNEX_OIDC_ALGORITHMS`   | `["RS256"]`                 | Signature algorithms accepted when verifying a token.          |
| `AUTOTUNEX_OIDC_EMAIL_CLAIMS` | `["email","emailAddress"]`  | Claim names checked, in order, to resolve the token to an email.|
| `AUTOTUNEX_OIDC_LEEWAY_SECONDS`| `30`                       | Clock-skew tolerance for `exp`/`nbf` checks.                   |

**Keep both entries in `AUTOTUNEX_OIDC_EMAIL_CLAIMS`.** The two-entry default
`["email", "emailAddress"]` is load-bearing: an issuer may populate only one of the two, and
which one can differ by token *kind* from the same issuer. In particular, a token-introspection
response — the verifier planned for the opaque-access-token case below — may carry
`emailAddress` but not `email`, so trimming it would break that path once it ships. Leave both
unless you are certain your issuer only ever emits one.

**Security model.** The token must carry a usable `aud`. The service always
verifies `aud`, `iss`, and `exp`, and it never infers the signature algorithm
from the token's own header — the accepted algorithms come only from
`AUTOTUNEX_OIDC_ALGORITHMS`. That makes `alg:none` and RS256/HS256 key-confusion
attacks structurally impossible. Audience is checked unconditionally, so a token
minted for a *different* application on the same issuer is rejected rather than
silently accepted. A token with no usable `aud` is a signal to distrust, not a
setting to work around — this API never accepts a credential it cannot scope to
itself.

**Opaque access tokens.** Some providers issue *access* tokens that are opaque
(not JWTs) and cannot be verified offline against a JWKS endpoint. If yours does,
you have two options:

- Send an **ID token** instead. An ID token is unambiguously a JWT, and its
  `aud` is the client id — set `AUTOTUNEX_OIDC_AUDIENCE` to the client id in that
  case. Note that ID tokens often live longer than access tokens, so a leaked
  one stays valid longer; weigh that for your deployment.
- Use a **token-introspection verifier**, which validates the token by calling
  the provider on each request. This is designed but **not yet implemented** —
  until it ships, an opaque-access-token deployment must send the ID token.

To tell which kind your provider issues, decode one real access token: a JWT has
three dot-separated base64url segments whose middle segment decodes to JSON
containing `aud`. If it does not, it is opaque.

---

## 4. Browser session (backend-for-frontend)

```
AUTOTUNEX_AUTH_PROVIDERS=["session"]
```

For a browser UI. The service acts as a backend-for-frontend (BFF): it runs the
OIDC authorization-code + PKCE flow server-side and hands the browser an
httpOnly session cookie. **No token ever reaches JavaScript.**

The flow uses these endpoints, plus two impersonation endpoints documented below (the
`assume` half admin-only):

| Endpoint            | Method | Purpose                                                                  |
| ------------------- | ------ | ------------------------------------------------------------------------ |
| `/auth/login`       | GET    | Starts an authorization-code + PKCE (`S256`) flow; redirects to the IdP. |
| `/auth/callback`    | GET    | Exchanges the code server-side, mints the httpOnly session cookie, then redirects (`302`) to `AUTOTUNEX_PUBLIC_BASE_URL`. |
| `/auth/me`          | GET    | Reports the current principal (the `Principal` body below).               |
| `/auth/logout`      | POST   | Clears the session cookie and the `autotunex_assume` impersonation overlay (and returns the IdP logout endpoint, if set); requires authentication. |

Every `/auth/*` route — the four above and the two impersonation endpoints below —
is mounted at the service **root**, outside the `/api/v1` resource prefix.

During `/auth/login` the `state` and PKCE verifier live in a short-lived signed,
httpOnly `oauth_flow` cookie (a 5-minute TTL) — never server memory — so the flow
works identically behind any number of workers. `/auth/callback` validates
`state`, exchanges the code for tokens (the client secret never reaches the
browser), verifies the returned ID token, and mints the session cookie.

Both `/auth/login` and `/auth/callback` are mounted unconditionally, so the routes
exist even when `"session"` is not among `AUTOTUNEX_AUTH_PROVIDERS`. Whether the
provider is disabled or merely configured incompletely, they answer with the same
opaque `401` as a rejected credential rather than a `404` — deliberately, so that
probing cannot distinguish the two cases.

### The `/auth/me` response

`GET /auth/me` returns the resolved principal verbatim — the same five fields
every other endpoint scopes itself by:

| Field          | Type           | Meaning                                                                              |
| -------------- | -------------- | ------------------------------------------------------------------------------------ |
| `email`        | string \| null | The verified email. Every shipped provider sets it — standalone included, which falls back to `standalone@autotunex.local`. |
| `provider`     | string         | Which credential kind verified the caller: `standalone`, `api_key`, `oidc`, or `session`. (Standalone reports `standalone`, not the `disabled` provider *setting* value.) |
| `user_id`      | UUID \| null   | The caller's `users` row id, or `null` when no row matches — the case just-in-time provisioning below addresses. |
| `is_admin`     | boolean        | Whether the caller may pass `?scope=all` and reach the admin-only endpoints.          |
| `impersonator` | string \| null | The real admin's email while an impersonation overlay is active; `null` otherwise.     |

### Required settings

| Setting                                | Meaning                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| `AUTOTUNEX_OIDC_ISSUER`                | The issuer (`iss`) of the ID token.                                                         |
| `AUTOTUNEX_OIDC_JWKS_URI`              | Where the issuer publishes its signing keys.                                                |
| `AUTOTUNEX_OIDC_AUDIENCE`              | The ID token's audience — **equal to the client id**.                                       |
| `AUTOTUNEX_OIDC_CLIENT_ID`             | The BFF's OIDC client id.                                                                    |
| `AUTOTUNEX_OIDC_CLIENT_SECRET`         | The BFF's OIDC client secret, exchanged server-side at the token endpoint.                  |
| `AUTOTUNEX_OIDC_AUTHORIZATION_ENDPOINT`| Where `/auth/login` redirects the browser.                                                  |
| `AUTOTUNEX_OIDC_TOKEN_ENDPOINT`        | Where `/auth/callback` exchanges the authorization code.                                    |
| `AUTOTUNEX_PUBLIC_BASE_URL`            | Public origin of this service, **no trailing slash** (see below).                           |
| `AUTOTUNEX_SESSION_SECRET`             | Signs the session cookie; **≥ 32 characters**, no random fallback (see below).              |

```
AUTOTUNEX_AUTH_PROVIDERS=["session"]
AUTOTUNEX_OIDC_ISSUER=https://idp.example.com
AUTOTUNEX_OIDC_JWKS_URI=https://idp.example.com/jwks
AUTOTUNEX_OIDC_AUDIENCE=<client-id>
AUTOTUNEX_OIDC_CLIENT_ID=<client-id>
AUTOTUNEX_OIDC_CLIENT_SECRET=<client-secret>
AUTOTUNEX_OIDC_AUTHORIZATION_ENDPOINT=https://idp.example.com/authorize
AUTOTUNEX_OIDC_TOKEN_ENDPOINT=https://idp.example.com/token
AUTOTUNEX_PUBLIC_BASE_URL=https://autotunex.example.com
AUTOTUNEX_SESSION_SECRET=<generate with secrets.token_urlsafe(32)>
```

Generate a session secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # yields 43 chars
```

The ID token's audience is the client id, so set `AUTOTUNEX_OIDC_AUDIENCE` to the
same value as `AUTOTUNEX_OIDC_CLIENT_ID`. If you enable `"oidc"` and `"session"`
together, both build their verifier from the one `AUTOTUNEX_OIDC_AUDIENCE`, so
set it to the client id in that combined case as well.

### Optional settings

| Setting                              | Default | Meaning                                                                 |
| ------------------------------------ | ------- | ----------------------------------------------------------------------- |
| `AUTOTUNEX_OIDC_END_SESSION_ENDPOINT`| unset   | The IdP's RP-initiated logout URL, returned by `/auth/logout` if set.   |
| `AUTOTUNEX_SESSION_TTL_HOURS`        | `8`     | How long a minted `session` cookie lasts, and how long an `autotunex_assume` impersonation overlay lasts (bounded `1`–`24`). |

### Deployment requirements

- **`AUTOTUNEX_PUBLIC_BASE_URL` must have no trailing slash.** The `redirect_uri`
  is built as `<public_base_url>/auth/callback`; a trailing slash produces a
  double slash the IdP will not recognize. `AUTOTUNEX_PUBLIC_BASE_URL` is used
  only from configuration — never taken from a request header.
- **That exact `redirect_uri` must be registered at your provider** for this
  client id, or the code exchange is rejected.
- **Cookies are always `Secure`** — serve behind TLS. A browser will not send a
  `Secure` cookie back over plain HTTP, so login silently never completes
  otherwise. (`localhost` is the exception: modern browsers treat it as a
  trustworthy origin even without TLS.)

### Cross-origin UIs

If the UI is served from a **different origin** than the API, set both of these —
they are required together, and startup refuses `same_site=none` without a CORS
allowlist:

```
AUTOTUNEX_SESSION_COOKIE_SAME_SITE=none
AUTOTUNEX_CORS_ALLOW_ORIGINS=["https://ui.example.com"]
```

The allowlist must **never contain `"*"`**: combined with the credentialed CORS a
BFF requires, a wildcard origin would let the service echo back any request's
`Origin` with the session cookie attached, defeating the allowlist entirely.
Startup refuses a `"*"` entry outright.

### Design decisions

Two omissions in this flow are deliberate — a decision, not an oversight:

- **No `nonce`.** Authorization-code + PKCE (`S256`) + `state` + the server-side token
  exchange already cover the ID-token-replay threat a `nonce` addresses; adding it on top
  would be belt-and-suspenders, not a gap.
- **The flow cookie stays at `path=/`, not scoped to `/auth`.** A narrower path would trim
  exposure slightly, but cookie deletion is matched by name, domain, and path alone — not by
  `Secure`/`SameSite` — so a narrower path would have to be threaded correctly through every
  place the cookie is cleared, or a deletion would silently miss and leave a stale flow cookie
  behind. The PKCE verifier inside that cookie is signed but **not encrypted** (anyone holding
  the cookie can base64-decode it), and that is intentional: `HttpOnly` keeps it away from
  JavaScript, and PKCE binds the authorization code to whoever holds the cookie — not to the
  verifier's bytes staying secret.

### Impersonation (assume / unassume)

Two further endpoints let an **admin** act as another user's data-owner identity —
the "assume user" control the UI drives. They are gated on the **real** caller,
so the control can never be exercised *through* an existing impersonation. The
overlay is not tied to the browser session, though: it requires only an admin
caller, a configured `AUTOTUNEX_SESSION_SECRET`, and the `autotunex_assume`
cookie, so it works with **any** credential kind — an API key or bearer token
included, not just a `"session"` login.

| Endpoint                 | Method | Purpose                                                              |
| ------------------------ | ------ | -------------------------------------------------------------------- |
| `/auth/assume/{user_id}` | POST   | Admin only — begin acting as `user_id`'s data-owner identity.        |
| `/auth/unassume`         | POST   | Drop the overlay, restoring the caller's own identity (idempotent).  |

Admin gating is enforced on the **real** caller (resolved by `get_principal`),
never the effective/assumed identity, so an admin cannot chain one impersonation
into another. The caller's own `is_admin` privileges are **preserved** while only
the data-owner identity switches — an admin keeps admin powers while acting as the
target. The overlay is carried in a signed, expiring `autotunex_assume` httpOnly
cookie, separate from the `session` cookie: the real login is never re-minted. Its
lifetime — both the token's `exp` and the cookie's `max-age` — comes from the same
`AUTOTUNEX_SESSION_TTL_HOURS` setting that bounds the session cookie.
`POST /auth/unassume` simply clears that cookie, and so does `POST /auth/logout`.

**`impersonator` is the only signal that an overlay is active.** While one is,
`email` and `user_id` on the principal are the *target's* effective identity and
`is_admin` is still the *real* admin's preserved flag — so nothing else in the
response distinguishes an impersonated caller from a genuine one. A UI that wants
to show an "acting as" banner must read `impersonator` from `GET /auth/me`. It is
`null` for an ordinary principal, and is only ever set on the effective principal
resolved per request — never on the real caller the admin gating checks.

On success each of the three POST endpoints returns `200` with a small JSON body:

| Endpoint                 | Response body                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `/auth/assume/{user_id}` | `{"message", "assumed_user_id", "assumed_email"}` — the target's id (as a string) and its email. |
| `/auth/unassume`         | `{"message"}` — the same shape whether or not an overlay was actually active (it is idempotent). |
| `/auth/logout`           | `{"end_session_endpoint"}` — the IdP's RP-initiated logout URL, for the client to redirect to.   |

`end_session_endpoint` is **always present**. When
`AUTOTUNEX_OIDC_END_SESSION_ENDPOINT` is unset it is `null` rather than absent, so
a client branches on the value and never has to probe for the key.

`POST /auth/assume/{user_id}` fails with:

| Status | Condition                                                                      |
| ------ | ------------------------------------------------------------------------------ |
| `403`  | The real caller is not an admin (`AdminRequiredError`).                        |
| `400`  | The target is the caller's own identity (`CannotImpersonateSelfError`).        |
| `503`  | No `AUTOTUNEX_SESSION_SECRET` is configured (`ImpersonationUnavailableError`). |
| `404`  | No user exists with that id (`UserNotFoundError`).                             |

---

## Just-in-time user provisioning

Real providers (API key, OIDC, session) verify an email but do not, by
themselves, create a `users` row for it. A caller with a verified email but no
matching row authenticates yet owns nothing — reads return an empty page, and a
*write* is refused with `403 Forbidden`.

Enable `AUTOTUNEX_AUTO_PROVISION_USERS=true` to create that row automatically on
the caller's first request, so a first-time caller owns what it creates. It is
off by default because it is an authorization *policy* and because it makes even
a `GET` write to the database on a caller's first request. Provisioned rows are
always `role='user'` (never admin). Standalone mode is unaffected — it always
resolves to a concrete owner that is provisioned regardless of this flag.

---

## See also

- [overview.md](overview.md) — pagination, ownership scoping, error shapes, status codes.
- [../operations/configuration.md](../operations/configuration.md) — the full settings reference.
- [../operations/deployment.md](../operations/deployment.md) — production requirements and the no-auth guard.
