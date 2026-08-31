#!/bin/sh
# Entrypoint for the all-in-one image (Dockerfile.aio).
#
# Materializes git credentials from the RUNTIME environment so that any git pull
# happening inside this container at run time — e.g. gbserver fetching the
# trainer repo during a build — can authenticate against a private host.
#
# Nothing is baked into the image: the host comes from $GITHUB_HOST and the
# token from $GB_TOKEN, both injected via --env-file at run time. The credential
# is written to an ephemeral file under /tmp (NOT the /data volume), so the token
# is not persisted; only the non-secret helper directive lands in git's config.
#
# Caveats:
#   * This authenticates *git* only. If the trainer is pulled by `pip` from a
#     private index, or a build runs in a separate executor/pod rather than in
#     this container, credentials must be wired into that path instead.
set -e

if [ -n "${GB_TOKEN:-}" ] && [ -n "${GITHUB_HOST:-}" ]; then
  cred_file=/tmp/aio-git-credentials
  # `store --file=` keeps the secret off the persistent volume; the --global
  # config only records the (non-secret) helper directive.
  git config --global credential.helper "store --file=${cred_file}"
  printf 'https://x-access-token:%s@%s\n' "${GB_TOKEN}" "${GITHUB_HOST}" > "${cred_file}"
  chmod 600 "${cred_file}"
  echo "aio-entrypoint: configured git credentials for host ${GITHUB_HOST}"
else
  echo "aio-entrypoint: GB_TOKEN/GITHUB_HOST not both set; skipping git credential setup"
fi

# Hand off to supervisord as PID 1 (runs all three services).
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/aio.conf
