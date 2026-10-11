"""Check CI orchestration contracts without application services or native builds."""

import json
import os
import re
import subprocess
import unittest
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def read_yaml(path: Path) -> dict[str, Any]:
    # BaseLoader preserves GitHub's `on` key instead of treating it as YAML 1.1 true.
    value = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


WORKFLOW = read_yaml(ROOT / ".github/workflows/ci.yml")
JOBS = WORKFLOW["jobs"]
NATIVE = read_yaml(ROOT / ".github/actions/setup-native/action.yml")
PYTHON = read_yaml(ROOT / ".github/actions/setup-python/action.yml")


def commands(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


class CIContracts(unittest.TestCase):
    def test_required_gate_covers_every_job_even_after_failure(self) -> None:
        # Given: Every non-gate job contributes required CI evidence.
        required = set(JOBS) - {"ci"}
        gate = JOBS["ci"]
        # When: GitHub schedules the aggregate check after its dependencies.
        self.assertEqual(gate["if"], "always()")
        self.assertEqual(set(gate["needs"]), required)
        # Then: No failed or skipped producer can bypass the original check name.
        self.assertEqual(gate.get("name", "ci"), "ci")
        for job in JOBS.values():
            self.assertNotIn("continue-on-error", job)
            for step in job.get("steps", []):
                self.assertNotIn("continue-on-error", step)

    def test_gate_executes_fail_closed_for_each_dependency(self) -> None:
        # Given: The exact shell command and environment wiring used by the gate.
        step = JOBS["ci"]["steps"][0]
        self.assertEqual(step["env"]["JOB_RESULTS"], "${{ toJSON(needs) }}")
        success = {name: {"result": "success"} for name in JOBS["ci"]["needs"]}
        cases: list[tuple[dict[str, Any], bool]] = [(success, True), ({}, False)]
        for name in success:
            for result in ("failure", "cancelled", "skipped"):
                cases.append(({**success, name: {"result": result}}, False))
        # When: All jobs succeed, or any one job fails, cancels, or is skipped.
        for results, expected in cases:
            with self.subTest(results=results):
                completed = subprocess.run(
                    ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
                    env={**os.environ, "JOB_RESULTS": json.dumps(results)},
                    capture_output=True,
                    text=True,
                    check=False,
                )
                # Then: Only a nonempty set of entirely successful jobs passes.
                self.assertEqual(completed.returncode == 0, expected, completed.stderr)

    def test_independent_checks_do_not_wait_for_native_work(self) -> None:
        # Given: Formatting, Python static checks, and UI checks need no helper.
        for name in ("rust-format", "python-static", "ui", "native-deps"):
            # When: Their workflow jobs are scheduled.
            self.assertNotIn("needs", JOBS[name])
            # Then: Native failure cannot prevent these checks from starting.
            self.assertNotIn("services", JOBS[name])
        for name in ("python-static", "ui"):
            self.assertEqual(JOBS[name]["strategy"]["fail-fast"], "false")

    def test_existing_check_commands_are_preserved(self) -> None:
        # Given: The serial workflow required Rust, Python, and all four UI checks.
        self.assertIn("cargo fmt", commands(JOBS["rust-format"]))
        self.assertIn("--all -- --check", commands(JOBS["rust-format"]))
        rust = commands(JOBS["rust-tests"])
        self.assertIn("cargo clippy", rust)
        self.assertIn("--all-targets --locked -- -D warnings", rust)
        self.assertIn("cargo test --manifest-path native/webrtc/Cargo.toml --locked", rust)
        # When: Static/UI commands move to independent matrix jobs.
        self.assertEqual(
            JOBS["python-static"]["strategy"]["matrix"]["check"],
            ["lint", "typecheck"],
        )
        self.assertEqual(
            JOBS["ui"]["strategy"]["matrix"]["check"],
            ["api:check", "lint", "test", "build"],
        )
        self.assertIn('make "$CHECK"', commands(JOBS["python-static"]))
        self.assertIn('pnpm --dir ui "$CHECK"', commands(JOBS["ui"]))
        # Then: Full backend coverage, release build, and static linkage still run.
        self.assertIn("make coverage", commands(JOBS["python-tests"]))
        self.assertIn("alembic -c alembic.ini upgrade head", commands(JOBS["python-tests"]))
        self.assertIn("make rust-build", commands(JOBS["native-build"]))
        self.assertIn(
            "ldd native/webrtc/target/release/homesec-webrtc",
            commands(JOBS["native-build"]),
        )

    def test_native_cache_is_saved_before_tests_and_is_not_required_for_correctness(self) -> None:
        # Given: Cold native dependency preparation is shared by two build lanes.
        steps = JOBS["native-deps"]["steps"]
        prepare = next(i for i, step in enumerate(steps) if step.get("id") == "prepare")
        save = next(
            i for i, step in enumerate(steps) if step.get("uses") == "actions/cache/save@v4"
        )
        # When: The existing wrapper checks/builds the pinned native installations.
        self.assertNotIn("if", steps[prepare])
        self.assertEqual(
            steps[prepare]["run"],
            "python3 native/webrtc/build_native.py -- cargo --version",
        )
        # Then: Saving happens immediately, independent of later build/test results.
        self.assertGreater(save, prepare)
        self.assertIn("steps.prepare.outcome == 'success'", steps[save]["if"])
        for name in ("rust-tests", "native-build"):
            self.assertEqual(JOBS[name]["needs"], "native-deps")
        cache = next(step for step in NATIVE["runs"]["steps"] if step.get("id") == "cache")
        self.assertNotIn("fail-on-cache-miss", cache["with"])
        self.assertNotIn("restore-keys", cache["with"])
        self.assertIn("fingerprint", cache["with"]["key"])
        self.assertIn("native/webrtc/build_native.py", cache["with"]["key"])

    def test_python_tests_receive_this_runs_executable_not_a_cached_binary(self) -> None:
        # Given: Backend tests retain their original real-helper environment.
        backend = JOBS["python-tests"]
        self.assertEqual(backend["needs"], "native-build")
        upload = next(
            step
            for step in JOBS["native-build"]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        )
        download = next(
            step for step in backend["steps"] if step.get("uses") == "actions/download-artifact@v4"
        )
        # When: The helper crosses the workflow job boundary.
        self.assertEqual(upload["with"]["name"], download["with"]["name"])
        self.assertIn("${{ github.sha }}", upload["with"]["name"])
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(upload["with"]["overwrite"], "true")
        self.assertNotIn("run-id", download["with"])
        # Then: The exact-run binary regains executable permission before pytest.
        script = commands(backend)
        self.assertLess(script.index("chmod +x"), script.index("make coverage"))
        self.assertIn("GITHUB_PATH", script)
        self.assertEqual(list(backend["services"]), ["postgres"])
        self.assertIn("Postgres did not become ready", script)

    def test_contract_lane_has_python_and_nested_uv_is_locked(self) -> None:
        # Given: API generation invokes uv rather than a live backend.
        steps = JOBS["ui"]["steps"]
        python = next(
            step for step in steps if step.get("uses") == "./.github/actions/setup-python"
        )
        # When: The four UI checks run independently.
        self.assertEqual(python["if"], "matrix.check == 'api:check'")
        # Then: Only the contract lane installs Python, with locked dependencies.
        self.assertEqual(WORKFLOW["env"]["UV_LOCKED"], "1")
        self.assertIn("uv lock --check", commands(PYTHON["runs"]))
        self.assertIn("uv sync --group dev --locked", commands(PYTHON["runs"]))

    def test_triggers_platform_and_local_actions_remain_valid(self) -> None:
        # Given: Every PR and default-branch push must still run the full workflow.
        self.assertEqual(WORKFLOW["on"]["push"]["branches"], ["main", "master"])
        self.assertEqual(WORKFLOW["on"]["pull_request"], "")
        self.assertEqual(WORKFLOW["permissions"], {"contents": "read"})
        self.assertEqual(WORKFLOW["concurrency"]["cancel-in-progress"], "true")
        # When: Jobs and composite actions are expanded on the same Ubuntu family.
        for job in JOBS.values():
            self.assertEqual(job["runs-on"], "ubuntu-24.04")
            for step in job["steps"]:
                uses = step.get("uses", "")
                if uses.startswith("./"):
                    self.assertTrue((ROOT / uses / "action.yml").is_file(), uses)
        # Then: Every checked-in shell block parses before a runner executes it.
        for job in [*JOBS.values(), NATIVE["runs"], PYTHON["runs"]]:
            for step in job["steps"]:
                if "run" in step:
                    script = re.sub(r"\$\{\{.*?\}\}", "ci-value", step["run"])
                    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


if __name__ == "__main__":
    unittest.main()
