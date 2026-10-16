# Testing authentication locally

A runbook for exercising each authentication provider against a running server.

This is **not** reference documentation — [`api/authentication.md`](api/authentication.md)
explains what each provider means and [`operations/configuration.md`](operations/configuration.md)
lists every setting, while [`SECURITY.md`](../SECURITY.md) covers the threat model. What follows
is the operational half: concrete values, the order to do things in, and the failure
modes that look like bugs but are the design working.

Every claim here was verified against a running server or the installed
libraries, not inferred. Where something is environment-specific it says so.

## Which provider do you want?

| Provider | For | Credential | Works in Swagger UI? |
| --- | --- | --- | --- |
| `disabled` | Local development with no auth at all (the default) | none | n/a — every caller is an admin but sees own-data by default; add `?scope=all` for the cross-user view |
| `api_key` | Machine callers: monitors, CI, the tuning pipeline | `X-API-Key` header | **Yes**, fully |
| `oidc` | CLI and service callers holding a W3ID token | `Authorization: Bearer` | **Yes**, fully |
| `session` | A browser UI | `session` cookie | Partly — see [Swagger UI](#swagger-ui-what-works-and-what-does-not) |

`AUTOTUNEX_AUTH_PROVIDERS` is a **JSON list**, so `AUTOTUNEX_AUTH_PROVIDERS=oidc`
raises a settings error. It needs `["oidc"]`. Providers combine
(`["api_key","oidc"]`), except that `"disabled"` cannot combine with anything.

Production refuses to start with `["disabled"]` unless
`AUTOTUNEX_ALLOW_INSECURE_NO_AUTH=true` is set, which logs a loud startup warning.

## `api_key`

There is no endpoint that issues a key and no admin UI — **you mint one
yourself**. The server never stores or sees a raw key: `AUTOTUNEX_API_KEYS` maps
a SHA-256 **digest** to an owner's email, and the verifier hashes whatever the
caller presents and compares digests with `hmac.compare_digest`.

### Mint one

The raw key must not reach your shell history or `ps` output, where any local
user can read it. Generating and hashing inside one process avoids both:

```bash
python3 -c "
import hashlib, secrets, subprocess
key = secrets.token_urlsafe(32)
subprocess.run(['pbcopy'], input=key.encode(), check=True)
print('digest -> AUTOTUNEX_API_KEYS:', hashlib.sha256(key.encode()).hexdigest())
print('raw key is on your clipboard. Store it in a password manager now —')
print('nothing can recover it later; the server only ever holds the digest.')
"
```

Drop the `subprocess` line and `print(key)` if you would rather see it on
screen, accepting that it lands in scrollback. On Linux replace `pbcopy` with
`xclip -selection clipboard` or `wl-copy`.

### Configure and use

```bash
export AUTOTUNEX_AUTH_PROVIDERS='["api_key"]'
export AUTOTUNEX_API_KEYS='{"<the-64-char-digest>": "owner@example.com"}'
make dev
```

```bash
curl -i -H "X-API-Key: <the-raw-key>" http://localhost:8000/api/v1/jobs
```

Startup refuses if `api_keys` is empty while `"api_key"` is enabled, and refuses
any entry that is not 64 lowercase hex characters. The error names the *email*
and never the offending value, so a raw key pasted there by mistake cannot leak
into a crash log.

The dict takes many entries, which is the intended shape: **one key per caller**.
That gives per-caller attribution in `jobs.user_id` and lets you revoke one key
without disturbing the others.

**Nothing rotates keys.** `api_keys` is a static settings map, so changing or
revoking one means editing the value and restarting the process.

**A key inherits the authority of the `users` row it maps to.** Mapping a service
key to an `admin` row grants it the *ability* to request the cross-user view with
`?scope=all`, not automatic cross-user visibility — by default it still sees only
its own rows. Map machine callers to their own non-admin rows unless you
specifically want that.

## `oidc` — bearer tokens

### The W3ID test issuer

> **IBM-internal example only.** `idp.example.com` throughout this section is
> a generic placeholder for IBM's internal W3ID test issuer host — it is not
> a real, reachable IdP. The operational pattern (two OPs on one host, an
> issuer that is not a prefix of its own endpoint paths) generalizes; the
> specific hostname and paths below do not.

Two OpenID providers live on `idp.example.com` and **the obvious one is the
wrong one**. Use the discovery endpoint named in your own client registration.
For a client registered against `/oidc/endpoint/default`:

| Setting | Value |
| --- | --- |
| `AUTOTUNEX_OIDC_ISSUER` | `https://idp.example.com/oidc/endpoint/default` |
| `AUTOTUNEX_OIDC_JWKS_URI` | `https://idp.example.com/v1.0/endpoint/default/jwks` |
| `AUTOTUNEX_OIDC_AUDIENCE` | your client id (the `id_token`'s `aud`) |

The issuer is **not** a prefix of the endpoint paths — normal for IBM Verify, but
you cannot derive either from the other. `authorize`, `token`, `introspect` and
`userinfo` all live under `/v1.0/endpoint/default/`.

### Send the `id_token`, not the access token

Verified against the live issuer on 2026-08-02: W3ID's **access token is opaque** —
151 characters, three dot-separated segments whose contents are *not* base64url
JSON. It cannot be verified offline at all. A segment count of three is therefore
not enough to identify a JWT.

So a caller must send the **`id_token`**, whose `aud` is the client id and which
`OidcBearerVerifier` verifies with no code change. The alternative is an
`IntrospectionVerifier` against the introspection endpoint, which is designed but
not built.

The trade-off to know: the `id_token` lives **2 hours** against the access
token's **5 minutes**.

### Get an `id_token`

`.superpowers/sdd/w3id-task6-authcode-probe.py` runs the authorization-code flow
and verifies the result against the live JWKS with the real verifier. It prints
no token to stdout by design; two flags route one somewhere usable:

```bash
export W3ID_CLIENT_ID='<your-client-id>'
read -rs -p 'client secret: ' W3ID_CLIENT_SECRET && export W3ID_CLIENT_SECRET && echo
.venv/bin/python .superpowers/sdd/w3id-task6-authcode-probe.py --clipboard
```

- `--clipboard` pipes it to `pbcopy`; nothing touches disk. Clear it afterwards
  with `pbcopy </dev/null`.
- `--out FILE` writes it `0600` for `curl -H "Authorization: Bearer $(cat FILE)"`.
  Refuses to overwrite. Delete it when done — an `id_token` is a live bearer
  credential for two hours.

**Port 8000 must be free**, because the probe's listener binds the exact redirect
URI registered for the client. If your own server is on 8000, move it with
`make dev PORT=8001`.

Note the script lives under `.superpowers/`, which is git-ignored — it is a local
helper, not a shipped tool.

### Configure and use

```bash
export AUTOTUNEX_AUTH_PROVIDERS='["oidc"]'
export AUTOTUNEX_OIDC_ISSUER='https://idp.example.com/oidc/endpoint/default'
export AUTOTUNEX_OIDC_JWKS_URI='https://idp.example.com/v1.0/endpoint/default/jwks'
export AUTOTUNEX_OIDC_AUDIENCE='<your-client-id>'
make dev
```

```bash
curl -i -H "Authorization: Bearer <the-id-token>" http://localhost:8000/api/v1/jobs
```

Startup refuses `"oidc"` unless all three are set, and names whichever is
missing. Empty and whitespace-only count as unset.

## `session` — the browser flow

### Prerequisite: register the callback URI

`redirect_uri` is built as `f"{public_base_url}/auth/callback"` and the path is
**hardcoded**. It is never taken from a request header, because
`X-Forwarded-Host` is attacker-controllable and a poisoned value would redirect
the authorization code to a host of the attacker's choosing.

So the IdP client must have **`http://localhost:8000/auth/callback`** registered.
If your registration lists a different path, the authorize request fails on
redirect_uri mismatch and no environment variable can fix it. Keep any existing
redirect URI as well if another tool depends on it — the token probe above does.

### Configure

```bash
export AUTOTUNEX_AUTH_PROVIDERS='["session"]'
export AUTOTUNEX_OIDC_ISSUER='https://idp.example.com/oidc/endpoint/default'
export AUTOTUNEX_OIDC_JWKS_URI='https://idp.example.com/v1.0/endpoint/default/jwks'
export AUTOTUNEX_OIDC_AUDIENCE='<your-client-id>'
export AUTOTUNEX_OIDC_CLIENT_ID='<your-client-id>'
read -rs -p 'client secret: ' s && export AUTOTUNEX_OIDC_CLIENT_SECRET="$s" && unset s && echo
export AUTOTUNEX_OIDC_AUTHORIZATION_ENDPOINT='https://idp.example.com/v1.0/endpoint/default/authorize'
export AUTOTUNEX_OIDC_TOKEN_ENDPOINT='https://idp.example.com/v1.0/endpoint/default/token'
export AUTOTUNEX_PUBLIC_BASE_URL='http://localhost:8000'
export AUTOTUNEX_SESSION_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
make dev
```

Startup requires **nine** settings for `"session"` and names whichever is
missing. `AUTOTUNEX_OIDC_END_SESSION_ENDPOINT` is the only optional one — set it
if you want `/auth/logout` to return an IdP logout URL.

`AUTOTUNEX_SESSION_SECRET` has a **32-character minimum** and no random fallback.
A random fallback would start up looking healthy and then fail every session
unpredictably across worker processes. `secrets.token_urlsafe(32)` yields 43
characters.

`AUTOTUNEX_PUBLIC_BASE_URL` must have **no trailing slash** — you would get
`//auth/callback` and a redirect_uri mismatch.

### Log in

Open this in a browser **address bar**:

```
http://localhost:8000/auth/login
```

It is a top-level navigation by design. The browser follows the 302 to the IdP,
you sign in, the IdP redirects back to `/auth/callback`, the server exchanges the
code server-side (so `client_secret` never reaches the browser), verifies the ID
token, and sets the `session` cookie.

Confirm with `http://localhost:8000/auth/me`.

### Cross-origin UI

If the UI is served from a different origin than the API:

```bash
export AUTOTUNEX_SESSION_COOKIE_SAME_SITE=none
export AUTOTUNEX_CORS_ALLOW_ORIGINS='["https://your-ui-origin"]'
```

Required together — startup refuses `same_site=none` with no allowlist, and
refuses `"*"` in the allowlist outright. A wildcard combined with the credentialed
CORS this design needs would let Starlette echo back any request's `Origin` with
the session cookie attached.

## Swagger UI: what works and what does not

`/docs` and `/openapi.json` need no credential, so they always load. All three
security schemes are published unconditionally — the set is deliberately *not*
narrowed to the enabled providers, because that would let anyone reading
`/openapi.json` enumerate a deployment's configured credential set.

**`ApiKeyAuth` and `BearerAuth` work fully.** Authorize → paste the raw key or
the token → every request carries the header. For `BearerAuth` paste the token
value only; Swagger adds the `Bearer ` prefix.

**`SessionCookieAuth`'s box is a decoy.** Typing into it does nothing: browsers
forbid JavaScript from setting the `Cookie` header, and the session cookie is
`HttpOnly` besides. It does work *ambiently* — log in at `/auth/login` first,
then open `/docs` in the same browser, and the cookie rides along on same-origin
requests (`Path=/`, `SameSite=lax`). Leave the Authorize dialog untouched.

**Never "Try it out" on `/auth/login`.** It returns a 302 to the IdP, Swagger's
`fetch` follows redirects, and the resulting cross-origin request has no CORS
headers for `localhost`, so the browser blocks it. You get `Failed to fetch` with
a CORS hint. That is the browser working correctly, not a server bug — use the
address bar.

**Never authorize `BearerAuth` and `ApiKeyAuth` at the same time.** Swagger sends
both headers and the router rejects two explicit credentials with **400**
`Provide exactly one credential.` — not a 401. Clear one. (A bearer token plus an
ambient session cookie is fine: the explicit credential wins.)

## Diagnostics

### A 200 with an empty job list is success, not failure

This is the single most common "it looks broken" moment. Authentication resolves
in two stages: the credential yields an email, then that email is looked up in
`users` to resolve `user_id` and `is_admin`. An **authenticated but unprovisioned**
caller — a valid credential whose email has no `users` row — resolves to
`user_id=null, is_admin=false` and sees an empty page, never an error.

`GET /auth/me` distinguishes the cases:

- email present, `user_id` non-null → fully provisioned
- email present, `user_id: null` → authenticated, no `users` row: insert one
- 401 → the credential itself was rejected

Rather than inserting the row by hand, `AUTOTUNEX_AUTO_PROVISION_USERS=true` creates
it on the caller's first request (race-safe, and always `role='user'` — never admin,
so an admin still needs the manual role change below).

Set that row's `role` to `admin` **and** pass `?scope=all` to see every job rather than
only its owner's — admin is the ability to ask for the cross-user view, and the parameter
unlocks it (a non-admin passing `scope=all` gets a 403). The comparison is case-sensitive,
so `Admin` or ` admin` resolves to non-admin silently.

### Reading a 401

Every rejection is logged server-side at `WARNING` with an actionable message,
deliberately: the client's 401 detail is fixed and opaque, so the operator has to
learn everything from the log. If a request 401s and you cannot see why, **read
the server output** — it will name the likely cause. Credential-expiry
rejections log at `INFO`, because they are routine and self-correcting, so they
are visible at the default log level; raising `AUTOTUNEX_LOG_LEVEL` above `INFO`
loses them.

### Common failures

| Symptom | Cause |
| --- | --- |
| `Failed to fetch` on `/auth/login` in Swagger | Expected. Use the browser address bar. |
| IdP rejects the authorize request | `redirect_uri` not registered — see the prerequisite above |
| 400 `Provide exactly one credential.` | Two explicit credentials sent; clear one in Authorize |
| 401 `The access token has expired.` | Re-mint. `id_token` lasts 2h, access token 5 min |
| 401 on every bearer token | Sending the opaque access token instead of the `id_token` |
| Login appears to succeed but no session | Cookie dropped — see the localhost note below |
| Settings error on startup | `AUTOTUNEX_AUTH_PROVIDERS` needs JSON: `["oidc"]`, not `oidc` |
| Probe hangs on "Listening…" | Port 8000 taken; move the server with `make dev PORT=8001` |

### `Secure` cookies over `http://localhost`

All three cookies — `oauth_flow`, `session` and `autotunex_assume` — are
unconditionally `Secure`. Chrome (≥89) and Firefox (≥75) treat
`http://localhost` as a trustworthy origin and will both set *and* send `Secure`
cookies there, so local testing works. Any **other** plain-`http` host will
silently drop them and login will appear to do nothing. Safari and older browsers
may also drop them on localhost.

Python's `http.cookiejar` has **no** localhost exception, which is why the test
suite drives these endpoints over `https://testserver`.

## Known gaps

- **No `https` scheme validation** on the URL settings. A plain-`http`
  `AUTOTUNEX_OIDC_TOKEN_ENDPOINT` would send `client_secret` and the
  authorization code in cleartext. Use `https` for anything but localhost.
- **No token introspection verifier**, so an access-token-only caller cannot use
  `"oidc"` against W3ID as it stands.
- **No CSRF tokens.** Every mutating endpoint relies on `samesite=lax` alone:
  `POST /jobs`, `POST /jobs/{id}/cancel`, `POST /jobs/{id}/reconcile` and
  `DELETE /jobs/{id}`; the full configuration and dataset CRUD, including `PUT`,
  `DELETE` and `POST /datasets/{id}/upload`; `PATCH /users/{id}`;
  `POST /auth/assume/{user_id}` and `POST /auth/unassume`; `POST /auth/logout`;
  and `POST /chat` / `POST /chat/stream`, whose tool registry can create a
  configuration and launch a job. Double-submit tokens across all mutating
  endpoints remain a tracked repo-wide change — see `CLAUDE.md`'s open decision 5.
- **API keys do not rotate**, as above.
