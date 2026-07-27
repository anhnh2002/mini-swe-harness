"""Frozen framework contract: bridges mini-swe-agent to the harbor task runner.

This file is NOT part of the edit space. It is the boundary between the
harness under optimization (the vendored ``minisweagent/`` package and
``terminal_bench.yaml``) and harbor — analogous to terminus-2's ``harbor.*``
imports. Optimization agents must never edit it.

harbor instantiates and drives :class:`MiniSweHarborAgent`. mini-swe-agent's
``DefaultAgent`` is synchronous and runs commands through a sync ``Environment``;
harbor hands us an *async* environment whose ``exec`` runs in the container.
We bridge the two by running the agent in a worker thread and marshalling each
command back onto harbor's event loop (:class:`HarborBridgeEnv`).
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import yaml
from harbor.agents.base import BaseAgent

import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import Submitted
from minisweagent.models.litellm_model import LitellmModel

_CONFIG_PATH = Path(__file__).with_name("terminal_bench.yaml")
_SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
# Consecutive command-marshaling failures (the ``except Exception`` path in
# ``HarborBridgeEnv.execute``) after which the environment is presumed dead.
_MAX_CONSECUTIVE_EXEC_FAILURES = 3


class TrialAborted(Exception):
    """The trial's container is gone (harbor cancelled the trial, or every
    recent exec failed to reach it); the agent loop must stop.

    Deliberately NOT a minisweagent flow-control exception: ``DefaultAgent.run``
    records it as a final ``exit_status: "TrialAborted"`` message, saves the
    trajectory once, and re-raises — which ends the worker thread. Without
    this, a thread orphaned by harbor's ``wait_for`` cancellation keeps
    talking to a deleted container until its own limits trip (tb2-006: up to
    24 minutes and ~18% of evaluator spend per run).
    """


def _load_config() -> dict[str, Any]:
    return yaml.safe_load(_CONFIG_PATH.read_text()) or {}


class HarborBridgeEnv:
    """A mini-swe-agent ``Environment`` backed by harbor's container.

    Lives on the agent's worker thread; ``execute`` hops each command onto the
    harbor event loop (still running on the main thread) via
    ``run_coroutine_threadsafe``. The ``Submitted`` sentinel handling mirrors
    mini-swe-agent's ``LocalEnvironment._check_finished`` so the agent's normal
    "echo the sentinel to finish" control flow works unchanged.
    """

    def __init__(self, henv: Any, loop: asyncio.AbstractEventLoop, *,
                 user: str | None = None, timeout: int = 300):
        self._henv = henv
        self._loop = loop
        self._user = user if user is not None else getattr(henv, "default_user", "root")
        self._timeout = timeout
        self._abort = threading.Event()
        self._consecutive_exec_failures = 0
        self.config = {"timeout": timeout, "user": self._user}

    def abort(self) -> None:
        """Make the next ``execute`` raise :class:`TrialAborted`. Called from
        the event-loop thread when harbor cancels the trial; thread-safe."""
        self._abort.set()

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        # Checked BEFORE the marshaling try/except so the catch-all below can
        # never convert an abort into one more dead observation.
        if self._abort.is_set():
            raise TrialAborted(
                "harbor cancelled the trial; the container is gone"
            )
        command = action.get("command", "")
        try:
            res = asyncio.run_coroutine_threadsafe(
                self._henv.exec(command=command, user=self._user), self._loop
            ).result(timeout=self._timeout)
            stdout = res.stdout or ""
            stderr = getattr(res, "stderr", "") or ""
            output = {
                "output": stdout + (("\n" + stderr) if stderr else ""),
                "returncode": res.return_code,
                "exception_info": "",
            }
            self._consecutive_exec_failures = 0
        except Exception as e:  # surface as an observation, never crash the loop
            # ... unless nothing has reached the container for a while: N
            # consecutive marshaling failures mean the environment is dead
            # (torn down, or the loop is gone) and feeding the model more
            # rc=-1 observations only burns tokens against a lost trial.
            self._consecutive_exec_failures += 1
            if self._consecutive_exec_failures >= _MAX_CONSECUTIVE_EXEC_FAILURES:
                raise TrialAborted(
                    f"{self._consecutive_exec_failures} consecutive command "
                    f"executions failed to reach the container (last: {e}); "
                    "presuming the environment is dead"
                ) from e
            output = {
                "output": str(e),
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == _SUBMIT_SENTINEL and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            })

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {**self.config, **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment_type": "harbor_adapter.HarborBridgeEnv"}}}


class MiniSweHarborAgent(BaseAgent):
    """harbor ``BaseAgent`` that runs mini-swe-agent's ``DefaultAgent`` on a task.

    harbor resolves this via ``--agent-import-path harbor_adapter:MiniSweHarborAgent``
    and calls ``setup`` then ``run``. The verifier reads the container state after
    ``run`` returns, so success does not depend on a return value — only on the
    side effects the agent performed via ``environment.exec``.
    """

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        # forward only the framework-contract positionals; ignore any extra
        # harbor kwargs (e.g. api_base) — the model reads OPENAI_API_BASE/KEY
        # from the env that the evaluator exports.
        super().__init__(logs_dir, model_name)
        if model_name is None:
            raise ValueError("model_name is required for mini-swe-agent")
        self._model_name = model_name
        self._cfg = _load_config()

    @staticmethod
    def name() -> str:
        return "mini-swe-agent"

    def version(self) -> str | None:
        return minisweagent.__version__

    async def setup(self, environment: Any) -> None:
        # No persistent session: harbor exec is one-shot per command, so there
        # is nothing to start (cf. terminus-2's tmux session).
        return None

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        loop = asyncio.get_running_loop()
        env_cfg = dict(self._cfg.get("environment") or {})
        bridge = HarborBridgeEnv(
            environment, loop,
            user=getattr(environment, "default_user", "root"),
            timeout=int(env_cfg.get("timeout", 300)),
        )
        model_cfg = {**(self._cfg.get("model") or {}), "model_name": self._model_name}
        model = LitellmModel(**model_cfg)
        agent_dir = Path(environment.trial_paths.agent_dir)
        agent = DefaultAgent(
            model, bridge,
            output_path=agent_dir / "trajectory.json",
            **(self._cfg.get("agent") or {}),
        )
        try:
            await asyncio.to_thread(agent.run, instruction)
        except BaseException:
            # BaseException so CancelledError lands here: harbor's wait_for
            # cancels this await but CANNOT kill the worker thread. Flag the
            # bridge so the orphan's next execute() raises TrialAborted and
            # the thread unwinds, instead of running against a deleted
            # container until its own limits trip.
            bridge.abort()
            raise
        finally:
            self._record_usage(context, agent)

    @staticmethod
    def _record_usage(context: Any, agent: Any) -> None:
        """Best-effort token/cost accounting. The verdict comes from harbor's
        verifier, not from these fields; they are only for logging/metrics."""
        try:
            context.cost_usd = agent.cost or None
        except Exception:
            pass
        n_in = n_out = 0
        for msg in getattr(agent, "messages", []):
            usage = ((msg.get("extra") or {}).get("response") or {}).get("usage") or {}
            n_in += usage.get("prompt_tokens") or 0
            n_out += usage.get("completion_tokens") or 0
        for attr, val in (("n_input_tokens", n_in), ("n_output_tokens", n_out),
                          ("n_cache_tokens", 0)):
            try:
                setattr(context, attr, val)
            except Exception:
                pass
