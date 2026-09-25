# Local stack operations (macOS)

Runtime entry points for the local-first deployment: the PATH wrapper every
caller goes through, the health monitor, and the watchdog LaunchAgent that
records stack health every 30 minutes. Together with `deploy/b550/` this is
the complete operational surface of a deployment; live files on the host
should be symlinks (scripts) or copies (plist) of this directory, so the
repo stays the single source of truth.

## Files

- `sindexer` — PATH wrapper (`~/.local/bin/sindexer`, with `rust-indexer`
  symlinked to it as a legacy alias). Sources `~/.context/.env` so shells,
  agents, and cron all get the same embedding endpoint and vector backend;
  variables already exported by the caller win over the profile. The binary
  itself reads only process env, so the wrapper is the only place the
  profile enters. It execs `/usr/local/bin/sindexer`, which points at this
  repo's release build.
- `sindexer-doctor` — health monitor (`~/.local/bin/sindexer-doctor`).
  Checks the full semantic-search path: wrapper, Jina embedding agent +
  endpoint dimension/latency, Milvus port + collections + container health,
  per-repo manifest freshness (7-day warn), LM Studio fallback, B550
  reachability, and volumes disk. Exit 0 healthy / 1 degraded / 2 broken.
  Modes: human report, `--json`, `--watch` (append one JSON line).
  A full cycle takes ~60s (the dead LM Studio port alone eats a 10s curl
  timeout); an in-flight run is not a hang.
- `launchd/com.local.sindexer-doctor.plist` — watchdog agent, one doctor
  `--watch` line every 30 minutes (StartInterval 1800, RunAtLoad) appended
  to `~/Library/Logs/sindexer-doctor.log`, pruned to 2000 lines.

## Install

```sh
repo=$(pwd)   # .../deploy/local
ln -sf "$repo/sindexer"         ~/.local/bin/sindexer
ln -sf "$repo/sindexer-doctor"  ~/.local/bin/sindexer-doctor
ln -sf sindexer                 ~/.local/bin/rust-indexer   # legacy alias
cp "$repo/launchd/com.local.sindexer-doctor.plist" ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.local.sindexer-doctor 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local.sindexer-doctor.plist
```

Scripts are symlinked so edits in the repo are live immediately; the plist
is copied because launchd parses it at load time — re-copy and re-bootstrap
after editing it.

## Conventions kept on purpose

- Both scripts pin `#!/bin/bash` (macOS system bash 3.2): it is the one
  interpreter guaranteed to exist in LaunchAgent and cron contexts, where
  Homebrew's tree is not on PATH. Keep them 3.2-clean (no associative
  arrays, `mapfile`, or `${var,,}`).
- The doctor self-augments PATH (Homebrew, `/usr/local/bin`, OrbStack) at
  the top: launchd's default PATH lacks `timeout` (coreutils) and `docker`
  (OrbStack), which otherwise silently degrades the collections and
  container checks into false "degraded" ticks.
- The doctor's REPOS list is the monitored fleet; extend it there when new
  repos are indexed.
