"""serve/test_launch_retry_gaps.py - the gaps of test_launch_retry.py: ram_oom_of's parsing of odd lines, ram_retry_gb's
boundaries (exact 0.5 steps, the 1.0 floor, the launcher's last try), the env keys, the log offset of a start, and
StrataEngine's restarts after a crash (restart()) over a scripted engine.

    python -m unittest serve.test_launch_retry_gaps -v
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
from serve.server import RAM_RETRY_FLOOR_GB, RAM_SET_KEYS, StrataEngine, ram_oom_of, ram_retry_gb  # noqa: E402

WIN = os.name == "nt"


def line(total: str, manual: str = "0") -> str:
    return f"glm fast: RAM-TIER-OOM total_gb={total} manual={manual}\n"


class ParseOddLines(unittest.TestCase):
    def test_not_matching(self):
        for s in [
            "glm fast: RAM-TIER-OOM total_gb=42.71 manual=0x\n",     # no word boundary after the flag
            "glm fast: RAM-TIER-OOM total_gb=42.71 manual=01\n",
            "glm fast: RAM-TIER-OOM total_gb=-1.00 manual=0\n",
            "glm fast: RAM-TIER-OOM total_gb=.5 manual=0\n",
            "glm fast: RAM-TIER-OOM total_gb=1e3 manual=0\n",
            "glm fast: RAM-TIER-OOM total_gb=42,71 manual=0\n",
            "glm fast: RAM-TIER-OOM total_gb=42. manual=0\n",
            "glm fast: RAM-TIER-OOM total_gb= manual=0\n",
            "glm fast: RAM-TIER-OOM total_gb=42.71\n",
            "glm fast: RAM-TIER-OOM manual=0 total_gb=42.71\n",     # fields in the other order
            "glm fast: RAM-TIER-OOM total_gb=42.71  manual=0\n",    # two spaces
            "glm fast: ram-tier-oom total_gb=42.71 manual=0\n",     # case
            "glm fast:RAM-TIER-OOM total_gb=42.71 manual=0\n",
        ]:
            self.assertIsNone(ram_oom_of(s), s)

    def test_matching_variants(self):
        self.assertEqual(ram_oom_of(line("42")), (42.0, False))                  # no decimals
        self.assertEqual(ram_oom_of(line("0.00", "1")), (0.0, True))
        self.assertEqual(ram_oom_of("glm fast: RAM-TIER-OOM total_gb=42.71 manual=0\r\n"), (42.71, False))   # CRLF
        self.assertEqual(ram_oom_of("[12:00] glm fast: RAM-TIER-OOM total_gb=42.71 manual=1."), (42.71, True))
        self.assertEqual(ram_oom_of("x" * 1000 + line("3.50") + "y" * 1000), (3.5, False))
        self.assertEqual(ram_oom_of(line("1024.00")), (1024.0, False))

    def test_last_valid_line_wins_over_a_later_garbled_one(self):
        self.assertEqual(ram_oom_of(line("42.71") + "glm fast: RAM-TIER-OOM total_gb=zz manual=0\n"), (42.71, False))

    def test_last_line_flag_wins(self):
        self.assertEqual(ram_oom_of(line("40.00", "1") + line("38.00", "0")), (38.0, False))
        self.assertEqual(ram_oom_of(line("40.00", "0") + line("38.00", "1")), (38.0, True))

    def test_two_lines_on_one_row(self):   # (a pipe that lost its newline)
        self.assertEqual(ram_oom_of(line("40.00").rstrip() + " " + line("38.00")), (38.0, False))


class RetryBoundaries(unittest.TestCase):
    def gb(self, total, manual="0", env=None, launcher=None, restarts=0):
        return ram_retry_gb(line(f"{total:.2f}", manual), env or {}, launcher, restarts)

    def test_exact_half_steps(self):
        # total - step lands exactly on a multiple of 0.5: no rounding loss (the 1e-9 slack, not one step lower)
        self.assertEqual(self.gb(40.0), 38.0)    # 5% = 2.0
        self.assertEqual(self.gb(30.0), 28.5)    # 5% = 1.5
        self.assertEqual(self.gb(60.0), 57.0)    # 5% = 3.0
        self.assertEqual(self.gb(50.0), 47.5)    # 5% = 2.5
        self.assertEqual(self.gb(10.0), 9.0)     # 1 GB step
        self.assertEqual(self.gb(3.0), 2.0)
        self.assertEqual(self.gb(2.5), 1.5)

    def test_just_below_a_half_step_rounds_down(self):
        self.assertEqual(self.gb(21.0), 19.5)    # 21 - 1.05 = 19.95
        self.assertEqual(self.gb(20.5), 19.0)    # 20.5 - 1.025 = 19.475
        self.assertEqual(self.gb(22.0), 20.5)    # 22 - 1.1 = 20.9
        self.assertEqual(self.gb(20.01), 19.0)   # 5% = 1.0005 > 1: 19.0095

    def test_twenty_gb_where_five_percent_is_one(self):
        self.assertEqual(self.gb(20.0), 19.0)
        self.assertEqual(self.gb(19.99), 18.5)   # 18.99 -> 18.5
        self.assertEqual(self.gb(20.02), 19.0)   # 5% = 1.001: 19.019 -> 19.0

    def test_near_the_floor(self):
        self.assertEqual(RAM_RETRY_FLOOR_GB, 1.0)
        self.assertEqual(self.gb(2.0), 1.0)
        self.assertEqual(self.gb(2.01), 1.0)
        self.assertEqual(self.gb(2.49), 1.0)
        self.assertIsNone(self.gb(1.99))         # 0.99 -> 0.5: under the floor
        self.assertIsNone(self.gb(1.5))
        self.assertIsNone(self.gb(1.0))
        self.assertIsNone(self.gb(0.5))
        self.assertIsNone(self.gb(0.0))

    def test_every_chain_ends_and_goes_down(self):
        for start in (42.71, 64.0, 20.0, 5.0, 2.0, 100.0, 1.2):
            t, steps = start, 0
            while True:
                nxt = self.gb(t)
                if nxt is None:
                    break
                self.assertLess(nxt, t)
                self.assertGreaterEqual(nxt, 1.0)
                t, steps = nxt, steps + 1
                self.assertLess(steps, 200)

    def test_launcher_try_caps_the_total(self):
        # the engine's line says more than the launcher asked (flooring): the step is taken from the launcher's try
        self.assertEqual(self.gb(45.0, "1", launcher=40.0, restarts=1), 38.0)
        # a line below the launcher's try: from the line
        self.assertEqual(self.gb(30.0, "1", launcher=40.0, restarts=1), 28.5)
        # manual=0 with a launcher try: the launcher's try still caps
        self.assertEqual(self.gb(45.0, "0", launcher=40.0, restarts=1), 38.0)

    def test_launcher_try_near_the_floor(self):
        self.assertEqual(self.gb(2.0, "1", launcher=2.0, restarts=1), 1.0)
        self.assertIsNone(self.gb(1.5, "1", launcher=1.5, restarts=1))
        self.assertIsNone(self.gb(9.0, "1", launcher=1.0, restarts=1))

    def test_launcher_try_does_not_override_the_users_setting(self):
        self.assertIsNone(self.gb(40.0, "1", env={"STRATA_GLM_RAM_GB": "40"}, launcher=40.0, restarts=1))

    def test_restart_cap(self):
        self.assertIsNone(ram_retry_gb(line("42.00"), {}, None, 3))
        self.assertIsNone(ram_retry_gb(line("42.00"), {}, None, 2, max_restarts=2))
        self.assertIsNotNone(ram_retry_gb(line("42.00"), {}, None, 2, max_restarts=3))
        self.assertIsNone(ram_retry_gb(line("42.00"), {}, None, 0, max_restarts=0))

    def test_none_or_empty_log(self):
        self.assertIsNone(ram_retry_gb(None, {}, None, 0))
        self.assertIsNone(ram_retry_gb("", {}, None, 0))


class EnvKeys(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(set(RAM_SET_KEYS), {"MAYA_GLM_RAM_GB", "STRATA_GLM_RAM_GB"})

    def test_each_name_alone_and_together(self):
        t = line("42.71")
        for env in ({"MAYA_GLM_RAM_GB": "30"}, {"STRATA_GLM_RAM_GB": "30"},
                    {"MAYA_GLM_RAM_GB": "30", "STRATA_GLM_RAM_GB": "40"}):
            self.assertIsNone(ram_retry_gb(t, env, None, 0), env)

    def test_one_empty_one_set(self):
        t = line("42.71")
        self.assertIsNone(ram_retry_gb(t, {"MAYA_GLM_RAM_GB": "", "STRATA_GLM_RAM_GB": "40"}, None, 0))
        self.assertIsNone(ram_retry_gb(t, {"MAYA_GLM_RAM_GB": "40", "STRATA_GLM_RAM_GB": ""}, None, 0))
        self.assertEqual(ram_retry_gb(t, {"MAYA_GLM_RAM_GB": "", "STRATA_GLM_RAM_GB": ""}, None, 0), 40.5)

    def test_zero_string_counts_as_set(self):   # (any non-empty value is the user's)
        self.assertIsNone(ram_retry_gb(line("42.71"), {"STRATA_GLM_RAM_GB": "0"}, None, 0))

    def test_other_variables_are_ignored(self):
        env = {"STRATA_GLM_RAM": "40", "MAYA_RAM_GB": "9", "PATH": "x"}
        self.assertEqual(ram_retry_gb(line("42.71"), env, None, 0), 40.5)


class LogSince(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "e.log"

    def since(self, offset, path=None, **kw):
        e = object.__new__(StrataEngine)
        e.log_path = str(self.path) if path is None else path
        return e._log_since(offset, **kw)

    def test_without_a_log(self):
        self.assertEqual(self.since(0, path=""), "")
        self.assertEqual(self.since(0), "")   # (not there)

    def test_from_the_offset_only(self):
        self.path.write_bytes(("old " + line("42.71") + "new\n").encode())
        off = len(("old " + line("42.71")).encode())
        self.assertEqual(self.since(off), "new\n")
        self.assertIsNone(ram_oom_of(self.since(off)))

    def test_offset_at_or_past_the_end(self):
        self.path.write_text("abc", encoding="utf-8")
        self.assertEqual(self.since(3), "")
        self.assertEqual(self.since(99), "")

    def test_only_the_last_bytes(self):
        self.path.write_bytes(b"A" * 1000 + b"B" * 100)
        self.assertEqual(self.since(0, most=100), "B" * 100)
        self.assertEqual(self.since(0, most=5000), "A" * 1000 + "B" * 100)

    def test_bad_utf8_is_replaced(self):
        self.path.write_bytes(b"ok \xff\xfe " + line("7.00").encode())
        self.assertEqual(ram_oom_of(self.since(0)), (7.0, False))


# a scripted engine like test_launch_retry's, but its limit is read from a file at every start (so it can change
# between a start and restart()) and it can print another failure: always (FAKE_PLAIN_FAIL), while the file
# FAKE_PLAIN_FILE exists, or when STRATA_GLM_RAM_GB is set (FAKE_PLAIN_FAIL_RAM: a restart fails for another reason).
# FAKE_MANUAL_FORCE=0|1: the manual flag it prints whatever STRATA_GLM_RAM_GB says (isolates the user-setting gate)
FAKE = textwrap.dedent(r"""
    import os, sys
    with open(os.environ["FAKE_STARTS"], "a") as f:
        f.write(os.environ.get("STRATA_GLM_RAM_GB", "-") + "\n")
    ram = os.environ.get("STRATA_GLM_RAM_GB") or None
    ok = float(open(os.environ["FAKE_OK_FILE"]).read())
    if (os.environ.get("FAKE_PLAIN_FAIL") or os.path.exists(os.environ.get("FAKE_PLAIN_FILE", "")) or
            (os.environ.get("FAKE_PLAIN_FAIL_RAM") and ram is not None)):
        sys.stderr.write("error: something else\n")
        sys.exit(1)
    if ram is None or float(ram) > ok:
        total = float(ram) if ram else float(os.environ["FAKE_AUTO_GB"])
        manual = os.environ.get("FAKE_MANUAL_FORCE") or (1 if ram else 0)
        sys.stderr.write("glm fast: RAM-TIER-OOM total_gb=%.2f manual=%s\n" % (total, manual))
        sys.exit(1)
    sys.stdout.write("READY 4096\n"); sys.stdout.flush()
    for ln in sys.stdin:
        if ln.startswith("QUIT"):
            break
""")


class EngineOverFake(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        script = d / "fake_engine.py"
        script.write_text(FAKE, encoding="utf-8")
        if WIN:
            self.exe = d / "fake_engine.bat"
            self.exe.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        else:
            self.exe = d / "fake_engine.sh"
            self.exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
            self.exe.chmod(0o755)
        self.log = d / "engine.log"
        self.starts = d / "starts.txt"
        self.okf = d / "ok.txt"
        self.okf.write_text("38")
        self.plainf = d / "plain-fail"   # (exists: every start fails for another reason)
        self.env = {k: v for k, v in os.environ.items() if k not in RAM_SET_KEYS}
        self.env.update(FAKE_STARTS=str(self.starts), FAKE_OK_FILE=str(self.okf), FAKE_AUTO_GB="42.71",
                        FAKE_PLAIN_FILE=str(self.plainf))
        p = mock.patch.object(server, "wait_gpu_release", lambda pid, timeout=60.0: None)
        p.start()
        self.addCleanup(p.stop)
        self.engines = []

    def tearDown(self):
        for e in self.engines:
            try:
                e.stop(timeout=10)
            except Exception:  # noqa: BLE001
                pass
            if getattr(e, "log", None) not in (None, server.subprocess.DEVNULL):
                e.log.close()
        for _ in range(50):
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
        return self.starts.read_text().split() if self.starts.exists() else []

    def test_restart_after_a_crash_starts_at_the_remembered_size(self):
        e = self.start()
        self.assertEqual(self.tries(), ["-", "40.5", "38"])
        e.restart()
        self.assertEqual(self.tries(), ["-", "40.5", "38", "38"])   # no climb back through 42.71
        self.assertEqual(e.max_context, 4096)
        self.assertEqual(e.ram_auto_gb, 38.0)

    def test_remembered_size_that_fails_again_steps_down_from_it(self):
        e = self.start()
        self.okf.write_text("34.5")
        e.restart()
        # 38 refused (manual=1 is the launcher's); 38 -> 36.1 -> 36.0 -> 34.2 -> 34.0 ready
        self.assertEqual(self.tries(), ["-", "40.5", "38", "38", "36", "34"])
        self.assertEqual(e.ram_auto_gb, 34.0)

    def test_restart_budget_is_per_start(self):
        e = self.start()
        self.okf.write_text("1")
        with self.assertRaises(RuntimeError):
            e.restart()
        # the restart began at 38 and had its own three: 38, 36, 34, 32
        self.assertEqual(self.tries()[3:], ["38", "36", "34", "32"])

    def test_first_start_without_the_oom_line_is_not_retried(self):
        with self.assertRaises(RuntimeError) as cm:
            self.start(FAKE_PLAIN_FAIL="1")
        self.assertEqual(self.tries(), ["-"])
        self.assertNotIn("restart(s)", str(cm.exception))

    def test_old_line_in_the_log_is_not_this_starts(self):
        self.log.write_text(line("42.71"), encoding="utf-8")   # a start of an earlier run
        with self.assertRaises(RuntimeError):
            self.start(FAKE_PLAIN_FAIL="1")
        self.assertEqual(self.tries(), ["-"])

    def test_error_names_restarts_and_last_try(self):
        self.okf.write_text("1")
        with self.assertRaises(RuntimeError) as cm:
            self.start()
        msg = str(cm.exception)
        self.assertIn("3 restart(s)", msg)
        self.assertIn("STRATA_GLM_RAM_GB=36", msg)
        self.assertIn("36.00 GB", msg)

    def test_floor_stops_the_restarts(self):
        self.okf.write_text("0.5")
        with self.assertRaises(RuntimeError):
            self.start(FAKE_AUTO_GB="2.0")
        self.assertEqual(self.tries(), ["-", "1"])   # 2.0 -> 1.0 (the floor runs), 1.0 -> none

    def test_total_under_two_gb_never_restarts(self):
        self.okf.write_text("0.5")
        with self.assertRaises(RuntimeError):
            self.start(FAKE_AUTO_GB="1.5")
        self.assertEqual(self.tries(), ["-"])

    def test_each_user_key_alone_blocks_the_restart(self):
        # (the engine says manual=0 here: the user's setting alone stands in the way of a restart)
        for k, first in (("MAYA_GLM_RAM_GB", "-"), ("STRATA_GLM_RAM_GB", "44")):
            with self.subTest(k):
                self.starts.unlink(missing_ok=True)
                with self.assertRaises(RuntimeError):
                    self.start(FAKE_MANUAL_FORCE="0", **{k: "44"})
                # (the engine only reads STRATA_GLM_RAM_GB here; MAYA_GLM_RAM_GB alone leaves it unset)
                self.assertEqual(self.tries(), [first])

    def test_empty_user_keys_do_not_block(self):
        e = self.start(MAYA_GLM_RAM_GB="", STRATA_GLM_RAM_GB="")
        self.assertEqual(e.max_context, 4096)
        self.assertEqual(self.starts.read_text().splitlines(), ["", "40.5", "38"])   # (the first: the empty setting)

    def test_os_environ_is_the_users_when_no_env_is_given(self):
        # MAYA_GLM_RAM_GB in the OS's environment only: the engine sees no size (manual=0), the env gate alone blocks
        with mock.patch.dict(os.environ, {**self.env, "MAYA_GLM_RAM_GB": "44"}, clear=True):
            with self.assertRaises(RuntimeError):
                StrataEngine(str(self.exe), [], log=str(self.log), env=None)
        self.assertEqual(self.tries(), ["-"])
        self.starts.unlink()
        with mock.patch.dict(os.environ, self.env, clear=True):   # (and without it the same start restarts)
            self.engines.append(StrataEngine(str(self.exe), [], log=str(self.log), env=None))
        self.assertEqual(self.tries(), ["-", "40.5", "38"])

    def test_without_a_log_there_is_no_restart(self):
        # (the RAM-TIER-OOM line is read from the log: no log, no line - documented; maya.py always gives a log)
        e = object.__new__(StrataEngine)
        self.engines.append(e)
        with self.assertRaises(RuntimeError) as cm:
            e.__init__(str(self.exe), [], log=None, env=self.env)
        self.assertEqual(str(cm.exception), "the engine exited before it was ready")
        self.assertEqual(self.tries(), ["-"])

    def test_restart_refused_for_another_reason_names_the_restarts_not_the_ram(self):
        with self.assertRaises(RuntimeError) as cm:
            self.start(FAKE_PLAIN_FAIL_RAM="1")
        self.assertEqual(self.tries(), ["-", "40.5"])
        msg = str(cm.exception)
        self.assertIn("1 restart(s) with a smaller RAM tier, the last at STRATA_GLM_RAM_GB=40.5", msg)
        self.assertNotIn("ran out of pinned RAM", msg)

    def test_remembered_size_then_an_unrelated_failure_names_no_restart(self):
        e = self.start()
        self.assertEqual(e.ram_auto_gb, 38.0)
        self.plainf.write_text("x")
        with self.assertRaises(RuntimeError) as cm:
            e.restart()
        self.assertEqual(self.tries()[3:], ["38"])
        self.assertEqual(str(cm.exception), f"the engine exited before it was ready (see {self.log})")

    def test_the_log_is_closed_before_the_error(self):
        e = object.__new__(StrataEngine)
        self.engines.append(e)
        with self.assertRaises(RuntimeError):
            e.__init__(str(self.exe), [], log=str(self.log), env={**self.env, "FAKE_PLAIN_FAIL": "1"})
        self.assertTrue(e.log.closed)


if __name__ == "__main__":
    unittest.main()
