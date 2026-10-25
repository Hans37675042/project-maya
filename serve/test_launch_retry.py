"""serve/test_launch_retry.py - a start that ran out of pinned RAM (the engine's `RAM-TIER-OOM` line) is started again
with a smaller STRATA_GLM_RAM_GB: the decision (ram_retry_gb) and StrataEngine's restarts over a scripted engine.

    python -m unittest serve.test_launch_retry -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server  # noqa: E402
from serve.server import RAM_RETRY_MAX, StrataEngine, ram_oom_of, ram_retry_gb  # noqa: E402

WIN = os.name == "nt"


def oom(total: float, manual: int = 0) -> str:
    return f"glm fast: CUDA1 RAM tier ...\nglm fast: RAM-TIER-OOM total_gb={total:.2f} manual={manual}\nerror: ...\n"


class OomLine(unittest.TestCase):
    def test_reads_total_and_manual(self):
        self.assertEqual(ram_oom_of(oom(42.71)), (42.71, False))
        self.assertEqual(ram_oom_of(oom(41.5, 1)), (41.5, True))

    def test_none_without_the_line(self):
        self.assertIsNone(ram_oom_of(""))
        self.assertIsNone(ram_oom_of("glm fast: CUDA0 warm-up: tables: cudaMemcpy failed (out of memory)\n"))
        self.assertIsNone(ram_oom_of(None))

    def test_the_last_line_wins(self):   # (a log the engine appends to: an older start's line above)
        self.assertEqual(ram_oom_of(oom(44.0) + oom(42.71)), (42.71, False))

    def test_garbled_line_is_none(self):
        self.assertIsNone(ram_oom_of("glm fast: RAM-TIER-OOM total_gb=abc manual=0\n"))
        self.assertIsNone(ram_oom_of("glm fast: RAM-TIER-OOM total_gb=42.0 manual=2\n"))


class RetryGb(unittest.TestCase):
    def test_no_line_no_retry(self):
        self.assertIsNone(ram_retry_gb("the engine said something else\n", {}, None, 0))

    def test_auto_line_retries_5pct_down_to_half_gb(self):
        # 42.71 - max(1, 2.1355) = 40.5745 -> 40.5
        self.assertEqual(ram_retry_gb(oom(42.71), {}, None, 0), 40.5)

    def test_small_tier_steps_one_gb(self):
        # 15.0 - max(1, 0.75) = 14.0 ; 10.3 - 1 = 9.3 -> 9.0
        self.assertEqual(ram_retry_gb(oom(15.0), {}, None, 0), 14.0)
        self.assertEqual(ram_retry_gb(oom(10.3), {}, None, 0), 9.0)

    def test_step_boundary_20gb(self):   # 5% of 20 = 1.0: either way one GB
        self.assertEqual(ram_retry_gb(oom(20.0), {}, None, 0), 19.0)

    def test_result_is_a_multiple_of_half_gb_and_smaller(self):
        for t in [x / 100 for x in range(150, 20000, 37)]:
            r = ram_retry_gb(oom(t), {}, None, 0)
            if r is None:
                continue
            self.assertEqual(r * 2, int(r * 2), t)
            self.assertLessEqual(r, round(t, 2) - max(1.0, round(t, 2) * 0.05) + 1e-9, t)
            self.assertGreater(r, round(t, 2) - max(1.0, round(t, 2) * 0.05) - 0.5, t)

    def test_floor_gives_up(self):
        self.assertIsNone(ram_retry_gb(oom(1.5), {}, None, 0))    # 0.5 GB: under the floor
        self.assertIsNone(ram_retry_gb(oom(0.0), {}, None, 0))
        self.assertEqual(ram_retry_gb(oom(2.0), {}, None, 0), 1.0)   # the floor itself still runs

    def test_manual_line_no_retry(self):
        self.assertIsNone(ram_retry_gb(oom(42.71, 1), {}, None, 0))

    def test_users_setting_no_retry(self):
        for k in ("MAYA_GLM_RAM_GB", "STRATA_GLM_RAM_GB"):
            self.assertIsNone(ram_retry_gb(oom(42.71), {k: "44"}, None, 0), k)
        self.assertEqual(ram_retry_gb(oom(42.71), {"MAYA_GLM_RAM_GB": ""}, None, 0), 40.5)   # empty: not set

    def test_launchers_own_setting_keeps_retrying(self):
        # the restart's STRATA_GLM_RAM_GB was the launcher's: the engine's manual=1 is that, not the user's
        self.assertEqual(ram_retry_gb(oom(40.5, 1), {}, 40.5, 1), 38.0)   # 40.5 - 2.025 = 38.475 -> 38.0

    def test_always_below_the_last_try(self):
        # an engine that reports more than the launcher asked for (floors, rounding) still goes down from the try
        self.assertEqual(ram_retry_gb(oom(41.0, 1), {}, 40.5, 1), 38.0)

    def test_restart_cap(self):
        self.assertEqual(RAM_RETRY_MAX, 3)
        self.assertIsNotNone(ram_retry_gb(oom(42.71), {}, None, RAM_RETRY_MAX - 1))
        self.assertIsNone(ram_retry_gb(oom(42.71), {}, None, RAM_RETRY_MAX))
        self.assertIsNone(ram_retry_gb(oom(42.71), {}, None, RAM_RETRY_MAX + 5))


# a scripted engine: refuses (the RAM-TIER-OOM line, exit 1) while STRATA_GLM_RAM_GB is unset or above FAKE_OK_GB,
# else READY; counts its starts in FAKE_STARTS
FAKE = textwrap.dedent(r"""
    import os, sys
    with open(os.environ["FAKE_STARTS"], "a") as f:
        f.write(os.environ.get("STRATA_GLM_RAM_GB", "-") + "\n")
    ram = os.environ.get("STRATA_GLM_RAM_GB")
    if ram is None or float(ram) > float(os.environ["FAKE_OK_GB"]):
        total = float(ram) if ram else float(os.environ["FAKE_AUTO_GB"])
        sys.stderr.write("glm fast: RAM-TIER-OOM total_gb=%.2f manual=%d\n" % (total, 1 if ram else int(os.environ.get("FAKE_MANUAL", "0"))))
        sys.stderr.write("error: glm fast: CUDA1 the pinned RAM tier did not allocate\n")
        sys.exit(1)
    sys.stdout.write("READY 4096\n"); sys.stdout.flush()
    for line in sys.stdin:
        if line.startswith("QUIT"):
            break
""")


class EngineRestarts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        script = d / "fake_oom_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        if WIN:
            self.exe = d / "fake_oom_engine.bat"
            self.exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        else:
            self.exe = d / "fake_oom_engine.sh"
            self.exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
            self.exe.chmod(0o755)
        self.log = d / "engine.log"
        self.starts = d / "starts.txt"
        self.env = {k: v for k, v in os.environ.items() if k not in ("MAYA_GLM_RAM_GB", "STRATA_GLM_RAM_GB")}
        self.env.update(FAKE_STARTS=str(self.starts), FAKE_AUTO_GB="42.71")
        patcher = mock.patch.object(server, "wait_gpu_release", lambda pid, timeout=60.0: None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.engines = []

    def tearDown(self):
        for e in self.engines:
            try:
                e.stop(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            if getattr(e, "log", None) not in (None, server.subprocess.DEVNULL):
                e.log.close()
        for _ in range(50):   # (Windows: the engine's log stays locked until the narrator and the process let go)
            try:
                self.tmp.cleanup()
                break
            except PermissionError:
                time.sleep(0.1)

    def start(self, **env):
        e = StrataEngine(str(self.exe), [], log=str(self.log), env={**self.env, **env})
        self.engines.append(e)
        return e

    def tries(self):
        return self.starts.read_text().split()

    def test_restarts_smaller_until_ready(self):
        e = self.start(FAKE_OK_GB="38")
        self.assertEqual(e.max_context, 4096)
        self.assertEqual(self.tries(), ["-", "40.5", "38"])   # 42.71 -> 40.5 -> 38.0
        self.assertEqual(e.ram_auto_gb, 38.0)

    def test_gives_up_after_three_restarts(self):
        with self.assertRaises(RuntimeError) as cm:
            self.start(FAKE_OK_GB="1")
        self.assertEqual(self.tries(), ["-", "40.5", "38", "36"])   # the first start and 3 restarts
        self.assertIn("exited before it was ready", str(cm.exception))
        self.assertIn("36", str(cm.exception))   # the last try's size goes with the error

    def test_manual_line_does_not_restart(self):
        with self.assertRaises(RuntimeError):
            self.start(FAKE_OK_GB="38", FAKE_MANUAL="1")
        self.assertEqual(self.tries(), ["-"])

    def test_users_setting_does_not_restart(self):
        with self.assertRaises(RuntimeError):
            self.start(FAKE_OK_GB="38", MAYA_GLM_RAM_GB="44", STRATA_GLM_RAM_GB="44")
        self.assertEqual(self.tries(), ["44"])

    def test_ready_engine_is_the_one_left_running(self):
        e = self.start(FAKE_OK_GB="38")
        self.assertTrue(e.alive())
        # every refused start ended before the next began: one process is left, the ready one
        self.assertEqual(len(self.tries()), 3)


if __name__ == "__main__":
    unittest.main()
