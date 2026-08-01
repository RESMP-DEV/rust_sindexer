# Examples

Copy-pasteable starting points for wiring `sindexer` into MCP clients and for
poking at the server without one.

## `mcp-servers.json`

An `mcpServers` configuration block with two variants:

- `code-indexer` — zero-config lexical-only mode (no env vars at all)
- `code-indexer-semantic` — semantic + lexical hybrid search via an
  OpenAI-compatible embedding provider

Where to put it, depending on your client:

| Client | Location |
| --- | --- |
| Claude Code (project) | `.mcp.json` in the repo root |
| Claude Code (user) | `~/.claude.json` under `mcpServers` (or `claude mcp add`) |
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) |
| Codex | `~/.codex/config.toml` (convert the JSON block to TOML) |

Replace `/absolute/path/to/sindexer` with the real path to the binary
(`target/release/sindexer` after `cargo build --release`).

## `demo.sh`

End-to-end smoke test that talks raw MCP JSON-RPC to the server over stdio —
no MCP client required. It:

1. creates a tiny throwaway Python project in a temp directory,
2. sends `initialize`, `index_codebase`, `search_code`, and `clear_index`
   requests as newline-delimited JSON,
3. prints every JSON-RPC response (pretty-printed if `jq` is installed).

```bash
./examples/demo.sh                      # auto-discovers ./target/release/sindexer
./examples/demo.sh /path/to/sindexer    # or pass the binary explicitly
```

Expected output: an `initialize` response, an `index_codebase` result with
`"success": true`, and a `search_code` result whose hits mention
`src/fib.py`. Useful for verifying a fresh build or a new platform port.
