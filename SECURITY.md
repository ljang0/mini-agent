# Security policy

Please report vulnerabilities privately to the repository maintainers rather
than opening a public issue. Include affected versions, a minimal reproduction,
and the expected impact.

`mini-agent` executes model-selected actions. Treat model output as untrusted:
use disposable workspaces or containers, keep credentials out of tool
environments, never mount a Docker socket into an agent container, and keep
benchmark verifiers and control-plane tokens outside the agent process.

The project supports Python 3.10–3.13. Security fixes target the latest released
minor version; older releases may not receive backports.
