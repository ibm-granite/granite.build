"use client";

/**
 * /auth/callback — GitHub redirects here after the user approves the OAuth app.
 *
 * URL params GitHub provides:
 *   ?code=<authorization_code>&state=<the_state_we_sent>
 *
 * This component:
 *   1. Reads code + state from the URL
 *   2. Retrieves the expected state nonce from sessionStorage (CSRF check)
 *   3. POSTs both to the sidecar, which exchanges the code for a token
 *   4. Stores the token + username in auth context
 *   5. Redirects to wherever the user was before they logged in
 */

import { useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { Loading } from "@carbon/react";
import { useAuth } from "@/auth/useAuth";
import axios from "axios";

const STATE_STORAGE_KEY = "gb-ui-oauth-state";
const REDIRECT_STORAGE_KEY = "gb-ui-pre-auth-url";

export default function AuthCallback() {
  const { login } = useAuth();
  const router = useRouter();
  const attempted = useRef(false);

  useEffect(() => {
    // Prevent double-execution in React StrictMode
    if (attempted.current) return;
    attempted.current = true;

    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    const state = params.get("state");
    const error = params.get("error");
    const expectedState = sessionStorage.getItem(STATE_STORAGE_KEY);
    const raw = sessionStorage.getItem(REDIRECT_STORAGE_KEY);
    const returnTo =
      !raw || raw === "/" || raw === "/login" ? "/dashboard" : raw;

    sessionStorage.removeItem(STATE_STORAGE_KEY);
    sessionStorage.removeItem(REDIRECT_STORAGE_KEY);

    // GitHub (or the sidecar callback) may relay an error param (e.g. access_denied).
    if (error) {
      router.push(`/login?error=${encodeURIComponent(error)}`);
      return;
    }

    if (!code || !state) {
      router.push("/login?error=missing_code");
      return;
    }

    if (!expectedState) {
      router.push("/login?error=missing_state");
      return;
    }

    axios
      .post("/api/analytics/auth/github/token", {
        code,
        state,
        expected_state: expectedState,
      })
      .then(({ data }) => {
        login(data.access_token, data.username, data.name);
        router.replace(returnTo);
      })
      .catch((err) => {
        const msg =
          err.response?.data?.detail ?? err.message ?? "Authentication failed";
        router.push(`/login?error=${encodeURIComponent(msg)}`);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
        background: "var(--cds-background)",
      }}
    >
      <Loading withOverlay={false} />
      <p
        style={{
          marginTop: "1rem",
          fontSize: "1rem",
          color: "var(--cds-text-primary)",
        }}
      >
        Completing sign in…
      </p>
    </div>
  );
}
