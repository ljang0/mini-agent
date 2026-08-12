# multi-mini-agent

The multi-agent front door for
[mini-agent](https://github.com/ljang0/mini-agent): the identical minimal
agent loop, with recursive delegation enabled by default.

`pip install multi-mini-agent` installs the core distribution
(`mini-agent-cmu`) and adds one console
script. `multi-mini-agent profile|run|eval ...` behaves exactly like
`mini-agent ... --multi-agent`: workers gain the single `agent` tool
(`spawn`, `send`, `inbox`, `wait`, `stop`, `adopt`), every worker remains an
ordinary `MiniAgent`, and the shared budget ledger, trace recorder, and
manifest contracts are unchanged. `grade` and `doctor` pass through untouched.

There is no separate multi-agent codebase — this package re-exports
`Orchestrator` and `CommunicationEnvironment` from `mini_agent` and injects
one flag. See the main repository for documentation, limits, and the security
policy.
