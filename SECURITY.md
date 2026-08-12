# Security policy

Report vulnerabilities privately via GitHub's
[private vulnerability reporting](https://github.com/ljang0/mini-agent/security/advisories/new)
rather than in a public issue. Include the affected version, a minimal
reproduction, expected impact, and whether secrets or benchmark-private data
may have been exposed. Expect an acknowledgment within a week.

`mini-agent` executes model-selected actions. Model output, repository contents,
web pages, screenshots, benchmark tasks, provider responses, and child-agent
messages are all untrusted input.

## Trust boundaries

- Direct and multi-agent local SWE runs execute arbitrary commands as the current
  user with host filesystem and network access. A worker's private repository copy
  and `HOME` isolate state from sibling workers; they are not a security sandbox.
  Use a benchmark container or a separately sandboxed compute node, and do not run
  over a home directory or credential store.
- Rootless Docker and Apptainer reduce host exposure but are not perfect security
  boundaries. Do not mount the Docker socket, SSH directories, cloud credentials,
  package credentials, or unrelated host paths into an agent environment. Apply
  host-level CPU, memory, process, disk, and network controls where required.
- The HTTP reader validates each redirect target against public DNS/IP ranges and
  bounds response size. DNS can change between validation and connection. The
  Playwright reader also executes active web content and cannot validate every
  subresource as a security boundary. Use network namespaces or an egress proxy
  when private-network access would be dangerous.
- A computer gateway URL and its bearer token can control a live machine. Bind
  gateways to a protected interface, use short-lived task-scoped tokens, require
  TLS for non-loopback deployments, and never reuse an environment lease across
  agents.
- Custom provider base URLs are trusted endpoints. Non-loopback endpoints must
  use HTTPS; literal loopback HTTP remains available for local inference. Do not
  embed credentials in URLs or request bodies, and use a dedicated environment
  variable through `--api-key-env`.
- The maintained OpenAI adapter uses provider-side `previous_response_id`
  continuation and therefore rejects `store: false`. Deployments that require Zero
  Data Retention need a downstream adapter that replays the complete response-item
  history.

## Secrets and private evaluation data

Traces redact known credential values, common secret-key fields, URL query tokens,
and image bodies. Redaction is defense in depth, not proof that arbitrary model or
tool text contains no secret. Keep credentials out of agent-visible files and
prompts.

Provider, SerpAPI, page-reader, and computer-gateway responses are streamed into
fixed byte limits before JSON/PNG decoding. Screenshots also have decoded-byte,
dimension, and pixel caps. Treat those limits as denial-of-service containment,
not validation that remote content is trustworthy.

Benchmark traces disable content capture. Even so, task artifacts may contain
model answers, patches, URLs, document references, screenshots, trajectories,
provider metadata, or hidden grader output. Keep private-answer artifacts,
qrels, verifier details, OSWorld screenshots, and proprietary task data out of
public validation bundles. Official grading snapshots hidden inputs and captured
grader output under its `0700` grade directory and hardens files to `0600`; these
permissions protect against other local users but do not make the contents safe
to publish or protect them from the account running the grader. Review every
artifact before publishing it.

Evaluation and resume artifacts are untrusted even when their plain SHA-256
fingerprints are internally consistent; those fingerprints detect accidental
change, not a malicious rewrite. Grading never executes a container-runtime
command selected by an evaluation manifest. SWE-bench image verification uses
the same Docker SDK contract, explicit endpoint, and environment as the upstream
grader. Official grader subprocesses receive only a narrow benchmark-specific
environment allowlist, not solver, browser, cloud, or computer-gateway
credentials inherited by the CLI.

Durable `MINI_AGENT_HOME` may contain run evidence indefinitely. Use a local
`MINI_AGENT_SCRATCH` for VM overlays and transient work, apply restrictive
filesystem permissions, and define a retention policy. Cleanup failures are run
failures; inspect and remove leaked containers, overlays, browser processes, or
VMs before another evaluation.

The project supports Python 3.10 through 3.13. Security fixes target the latest
release; older releases may not receive backports.
