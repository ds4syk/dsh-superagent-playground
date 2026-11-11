#!/bin/sh
# Create the merge CLI wrapper and verify that it is signed in as the reviewer.
#
# The wrapper uses its own gh configuration directory, so it never shares the
# agent's token (`~/.config/gh-agent`). Signing in is an operator step: the token
# must be pasted in the operator's own terminal, never in a chat.
#
# Usage:
#   scripts/merge-cli/setup-gh-merge.sh          # create the wrapper, check login
#   scripts/merge-cli/setup-gh-merge.sh --check  # check only, create nothing
#
# Sign in (one time, in the operator's terminal):
#   read -rs TOKEN          # paste the fine-grained token, then press Enter
#   printf '%s\n' "$TOKEN" | ~/.local/bin/gh-merge auth login --hostname github.com --with-token
#   unset TOKEN
set -eu

REVIEWER="YingkeSu"
AGENT="ds4syk"
WRAPPER="${HOME}/.local/bin/gh-merge"
CONFIG_DIR="${HOME}/.config/gh-merge"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

if [ "$CHECK_ONLY" -eq 0 ] && [ ! -x "$WRAPPER" ]; then
  mkdir -p "${HOME}/.local/bin"
  cat > "$WRAPPER" <<'EOF'
#!/bin/sh
GH_CONFIG_DIR="$HOME/.config/gh-merge" exec gh "$@"
EOF
  chmod +x "$WRAPPER"
  echo "created $WRAPPER"
fi

if [ ! -x "$WRAPPER" ]; then
  echo "missing $WRAPPER; run without --check first" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR"
chmod 700 "$CONFIG_DIR" 2>/dev/null || true
chmod 600 "$CONFIG_DIR/hosts.yml" 2>/dev/null || true

login=$("$WRAPPER" api user --jq .login 2>/dev/null || true)
if [ -z "$login" ]; then
  echo "gh-merge is not signed in." >&2
  echo "In your own terminal run:" >&2
  echo "  read -rs TOKEN; printf '%s\\n' \"\$TOKEN\" | $WRAPPER auth login --hostname github.com --with-token; unset TOKEN" >&2
  exit 1
fi

echo "gh-merge account: $login"
if [ "$login" != "$REVIEWER" ]; then
  echo "expected $REVIEWER (the reviewer), got $login" >&2
  exit 1
fi
if [ "$login" = "$AGENT" ]; then
  echo "refusing to use the agent account ($AGENT) for merging" >&2
  exit 1
fi

agent_login=$(gh-agent api user --jq .login 2>/dev/null || true)
echo "gh-agent account: ${agent_login:-unknown}"
[ "$agent_login" = "$AGENT" ] || echo "warning: gh-agent is not $AGENT" >&2

echo "ok: $WRAPPER is signed in as $REVIEWER"
