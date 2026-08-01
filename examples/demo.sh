#!/usr/bin/env bash
# End-to-end smoke test for the sindexer MCP server over raw stdio JSON-RPC.
# Creates a tiny throwaway project, indexes it, searches it, and clears the
# index — no MCP client required.
#
# Usage: examples/demo.sh [/path/to/sindexer]
set -euo pipefail

BIN="${1:-}"
if [[ -z "$BIN" ]]; then
  for candidate in ./target/release/sindexer ./target/debug/sindexer; do
    if [[ -x "$candidate" ]]; then
      BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$BIN" ]] && command -v sindexer >/dev/null 2>&1; then
  BIN="sindexer"
fi
if [[ -z "$BIN" ]]; then
  echo "error: sindexer binary not found" >&2
  echo "hint: build it with 'cargo build --release' or pass the path:" >&2
  echo "      examples/demo.sh /path/to/sindexer" >&2
  exit 1
fi

DEMO_DIR="$(mktemp -d)"
trap 'rm -rf "$DEMO_DIR"' EXIT

mkdir -p "$DEMO_DIR/src"
cat > "$DEMO_DIR/src/fib.py" <<'EOF'
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
EOF

echo "# binary:  $BIN" >&2
echo "# project: $DEMO_DIR" >&2

OUT_FILTER=(cat)
if command -v jq >/dev/null 2>&1; then
  OUT_FILTER=(jq -c .)
fi

{
  printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"sindexer-demo","version":"0.1.0"}}}'
  sleep 1
  printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/initialized"}'
  printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"index_codebase\",\"arguments\":{\"path\":\"$DEMO_DIR\"}}}"
  sleep 3
  printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"search_code\",\"arguments\":{\"path\":\"$DEMO_DIR\",\"query\":\"fibonacci number\",\"limit\":3}}}"
  sleep 2
  printf '%s\n' "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"clear_index\",\"arguments\":{\"path\":\"$DEMO_DIR\"}}}"
  sleep 1
} | "$BIN" 2>/dev/null | "${OUT_FILTER[@]}"
