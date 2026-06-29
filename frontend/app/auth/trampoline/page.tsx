"use client";

/**
 * /auth/trampoline — registered as the GitHub App callback URL.
 *
 * When running inside the OAuth popup (window.opener is set), it posts the
 * code + state back to the main window via postMessage and closes itself.
 * The main window then navigates to /auth/callback to complete the exchange.
 *
 * When reached via a full-page redirect (popup was blocked), it forwards
 * directly to /auth/callback as a normal navigation.
 */

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Loading } from "@carbon/react";
import { GITHUB_POPUP_MESSAGE_TYPE } from "@/auth/github";

export default function AuthTrampoline() {
  const router = useRouter();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    const state = params.get("state");
    const error = params.get("error");

    if (window.opener && window.opener !== window) {
      // Popup mode — relay back to the opener and close
      window.opener.postMessage(
        { type: GITHUB_POPUP_MESSAGE_TYPE, code, state, error },
        window.location.origin,
      );
      window.close();
      return;
    }

    // Full-page fallback (popup was blocked)
    if (error) {
      router.replace(`/login?error=${encodeURIComponent(error)}`);
    } else if (code && state) {
      router.replace(
        `/auth/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`,
      );
    } else {
      router.replace("/login?error=missing_code");
    }
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
