# Local stack operations (macOS)

Runtime entry points for the local-first deployment: the PATH wrapper every
caller goes through, the health monitor, and the watchdog LaunchAgent that
records stack health every 30 minutes. Together with `deploy/strix/` this is
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
  Checks the full semantic-search path: wrapper, persistent Strix CUDA
  embedding tunnel + endpoint model/runtime/device/dimension, Milvus port +
  collections + container health, per-repo manifest freshness (7-day warn),
  LM Studio fallback, Strix reachability, and volumes disk. Exit 0 healthy /
  1 degraded / 2 broken.
  Modes: human report, `--json`, `--watch` (append one JSON line).
  A full cycle takes ~60s (the dead LM Studio port alone eats a 10s curl
  timeout); an in-flight run is not a hang.
- `sindexer-strix-rebuild` — explicit full-rebuild entry point. Starts/checks
  the Strix RTX 5070 CUDA service, restores the Mac tunnel if needed, starts
  the local Milvus containers when OrbStack is stopped, holds a transient
  Strix sleep inhibitor, and runs `sindexer index <repo> --force` with a
  32-chunk embedding batch. Use `--update` for the same preflight with
  incremental update semantics.
- `launchd/com.local.sindexer-doctor.plist` — watchdog agent, one doctor
  `--watch` line every 30 minutes (StartInterval 1800, RunAtLoad) appended
  to `~/Library/Logs/sindexer-doctor.log`, pruned to 2000 lines.

## Install

```sh
repo=$(pwd)   # .../deploy/local
ln -sf "$repo/sindexer"         ~/.local/bin/sindexer
ln -sf "$repo/sindexer-doctor"  ~/.local/bin/sindexer-doctor
ln -sf "$repo/sindexer-strix-rebuild" ~/.local/bin/sindexer-strix-rebuild
ln -sf sindexer                 ~/.local/bin/rust-indexer   # legacy alias
cp "$repo/launchd/com.local.sindexer-doctor.plist" ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.local.sindexer-doctor 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local.sindexer-doctor.plist
```

Scripts are symlinked so edits in the repo are live immediately; the plist
is copied because launchd parses it at load time — re-copy and re-bootstrap
after editing it.

## Conventions kept on purpose

- Paths are host-specific by design: launchd `ProgramArguments` cannot
  expand `$HOME`, so the plist carries absolute paths and must be edited
  for a different user or machine before install. The doctor's REPOS list
  is `$HOME`-relative but describes this fleet's layout; adjust it for a
  different checkout topology.
- Local scripts use `#!/usr/bin/env bash`, explicitly restore PATH for
  sanitized LaunchAgent/cron contexts, and remain Bash 3.2-compatible (no
  associative arrays, `mapfile`, or `${var,,}`).
- The doctor self-augments PATH (Homebrew, `/usr/local/bin`, OrbStack) at
  the top: launchd's default PATH lacks `timeout` (coreutils) and `docker`
  (OrbStack), which otherwise silently degrades the collections and
  container checks into false "degraded" ticks.
- The doctor's REPOS list is the monitored fleet; extend it there when new
  repos are indexed.
