# mini-swe-harness

The agent harness under optimization: a vendored copy of
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (MIT) as a
standalone top-level package `minisweagent/`, plus a thin harbor adapter that
lets it run against Terminal-Bench 2 without modifying the agent itself.

Run against Terminal-Bench 2 with:

    PYTHONPATH=. harbor run -d terminal-bench@2.0 \
        --agent-import-path harbor_adapter:MiniSweHarborAgent \
        -m <litellm-model-id> -i <task> -o /tmp/tb2-check -n 1 -y

(harbor must be installed with mini-swe-agent's runtime deps — see the project
runbook §2.2.)

## Edit-space contract

The repo **is** the edit space. Exactly one mechanism per edit (single-factor
experiments only).

**Editable — the harness:**
- `terminal_bench.yaml` — the prompt templates (system / instance / observation /
  format-error) and the step/cost/timeout limits. This is the primary surface.
- `minisweagent/agents/default.py` — the turn loop: step/cost/wall-time limits,
  format-error handling, when the agent decides it is done.
- `minisweagent/models/litellm_model.py` — the model call: retry/backoff policy,
  cost tracking, response/tool-call parsing, observation formatting.
- the rest of `minisweagent/` (models utils, exceptions, serialization) is part
  of the harness and may be edited when a change genuinely belongs there.

**Frozen — the framework contract (never edit):**
- `harbor_adapter.py` — the harbor `BaseAgent` boundary and the sync↔async
  command bridge. Changing it changes how harbor talks to the harness, not the
  harness itself.
- The import path `harbor_adapter:MiniSweHarborAgent` must stay importable.
- Anything imported from `harbor.*` — that is the host framework's contract.
