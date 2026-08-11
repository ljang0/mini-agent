# Archived Scaffold Lab research

This directory preserves the pre-0.3 inference harnesses, topology studies,
configs, benchmarks, and deterministic tests as audited research material. It is
not installed by the `mini-agent` distribution and is not a second supported
agent framework.

The preservation tag `scaffoldlab-v0.2-handoff` remains the authoritative source
for exact historical reproduction. `mini_agent.research_catalog` contains the
read-only 18-lab audit metadata used by the current CLI. Explicit reference runs
may load this archive lazily from a source checkout; normal `mini-agent` imports,
runs, and wheels do not import `scaffoldlab`.

The original documentation is preserved in [`docs`](docs). Links and commands
inside those files describe the archived layout and may not apply to the 0.3
package.
