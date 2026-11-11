#!/usr/bin/env python3
"""Patch the deployment's Delivery requirements prompt with the merge rule.

Reads the current prompt configuration from the running dsh-superagent panel,
replaces the `<merge CLI>` placeholder with the wrapper path, appends the
approval/auto-merge rule, validates the result with the panel's preview route,
and only saves it with --apply.

This is deliberately a dry run by default: the live prompt changes what every
future worker is told, so inspect the diff and the composed section first.

Usage:
  python3 scripts/merge-cli/apply-prompt.py                 # show diff + preview
  python3 scripts/merge-cli/apply-prompt.py --apply         # save a new revision
  python3 scripts/merge-cli/apply-prompt.py --api http://127.0.0.1:3092/orchestrator/api
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
import urllib.error
import urllib.request

MERGE_CLI = "/home/syk/.local/bin/gh-merge"
REVIEWER = "YingkeSu"
REVIEWER_ID = "264377619"
MARK = "**Merge rule for this deployment:**"
BLOCK = (
    f"{MARK} The merge CLI is `{MERGE_CLI}`, signed in as @{REVIEWER} "
    f"(GitHub id {REVIEWER_ID}), never as the agent account. Request @{REVIEWER}'s review on every pull request you open. "
    f"Every pull request needs one approving review from @{REVIEWER} that covers its current head commit; a bot review, "
    "your own review, or an approval of an earlier commit does not count. This includes pull requests that change what "
    "users see or read, such as UI, rendered documents and published artifacts.\n\n"
    f"Because this deployment fixes the reviewer, credit @{REVIEWER} when you open the pull request rather than after "
    f"approval: put `Co-authored-by: {REVIEWER} <{REVIEWER_ID}+{REVIEWER}@users.noreply.github.com>` on every commit of "
    f"the branch, then request @{REVIEWER}'s review. Do not rewrite the branch after @{REVIEWER} approves it; a rewrite "
    "changes the head commit and GitHub dismisses the approval. If a later push is needed, add the trailer to the new "
    "commits too, withdraw auto-merge first, and request review again.\n\n"
    "For this deployment, a pull request that only adds or edits documentation, research notes, knowledge-base pages, "
    "images or generated documents, with no executable code, configuration or scripts, does not count against the "
    "200-line limit; keep the limit for code, automation, configuration and scripts."
)


def call(api: str, method: str, path: str, body: object | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        api + path,
        data=data,
        method=method,
        headers={"x-orchestrator-request": "1", "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"{method} {path} failed: HTTP {error.code}\n{detail}") from error


def patched(text: str) -> str:
    if MARK in text:
        raise SystemExit("the delivery section already contains the merge rule")
    if "<merge CLI>" not in text:
        raise SystemExit("the delivery section has no <merge CLI> placeholder; inspect it before patching")
    return text.replace("<merge CLI>", MERGE_CLI) + "\n\n" + BLOCK + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:3092/orchestrator/api")
    parser.add_argument("--apply", action="store_true", help="save the patched revision")
    parser.add_argument("--out", help="also write the patched config JSON here")
    args = parser.parse_args()

    view = call(args.api, "GET", "/prompts")
    revision = view["current"]["revision"]
    config = view["current"]["config"]
    old = config["delivery"]
    new = patched(old)

    print(f"current revision: {revision} (source: {view['current']['source']})")
    print(f"delivery: {len(old)} -> {len(new)} chars")
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), "delivery (current)", "delivery (patched)", lineterm=""):
        print(line)

    candidate = {**config, "delivery": new}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(candidate, handle, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")

    preview = call(args.api, "POST", "/prompts/preview", candidate)
    if MARK not in preview.get("worker", ""):
        raise SystemExit("preview does not contain the merge rule; refusing to save")
    print("preview: ok, the composed worker prompt contains the merge rule")

    if not args.apply:
        print("dry run: nothing saved. Re-run with --apply to save a new revision.")
        return 0

    saved = call(args.api, "PUT", "/prompts", {"config": candidate, "baseRevision": revision})
    print(f"saved revision: {saved['revision']} (context delivery: {saved.get('context_delivery')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
