"""scripts/gpu_record.py with hand-written logs and stand-in tools (no GPU, no llvm-mc, no game data).

    python -I tests/test_gpu_record.py

The logs are written here in the shape run_gpu_tests.py prints; `vulkaninfo`, `git` and the tool versions are
replaced by fixed facts.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import gpu_record as rec  # noqa: E402

NATIVE = """\
12 shaders: 12 translated, 12 pass spirv-val
device: Vendor One 9000 (subgroup 32..32)
  PASS  cs_add
  FAIL  cs_atomic            6 of 64 sint words differ; first word 32: got 0x0badf00d, expected 0x00000101
  SKIP  cs_w64_a             SKIP (no 64-wide compute subgroup on this device)
  SKIP  cs_w64_b             SKIP (no 64-wide compute subgroup on this device)
  SKIP  ps_draw              SKIP (run by d3d12_m3_frame_test)
  PASS  cs_mul
6/10 passed, 3 skipped
"""
EMULATED = """\
12 shaders: 12 translated, 12 pass spirv-val
device: Vendor One 9000 (subgroup 32..32)
  PASS  cs_add
  FAIL  cs_atomic            6 of 64 sint words differ; first word 32: got 0x0badf00d, expected 0x00000101
  FAIL  cs_w64_b             exit 1: lane 40 differs
  SKIP  ps_draw              SKIP (run by d3d12_m3_frame_test)
7/10 passed, 1 skipped
"""
VULKANINFO = """\
Devices:
========
GPU0:
\tapiVersion         = 1.4.351
\tdeviceName         = Vendor One 9000
\tdriverName         = VENDOR
\tdriverInfo         = 617.42
GPU1:
\tapiVersion         = 1.4.315
\tdeviceName         = Vendor Two Graphics
\tdriverInfo         = 25.10.36
"""
FACTS = {"driver": "VENDOR 617.42", "api": "1.4.351", "os": "Windows 11, 10.0.1", "commit": "a" * 40,
         "date": "2026-11-26", "tools": ["llvm-mc 21.1.8", "Python 3.14.7"]}


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(rec, "machine", lambda device, llvm_mc=None: FACTS), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rec.main(argv)
    return code, out.getvalue(), err.getvalue()


class ParseLog(unittest.TestCase):
    def test_counts_device_and_shaders(self):
        run = rec.parse_log(NATIVE)
        self.assertEqual(run["device"], "Vendor One 9000")
        self.assertEqual(run["subgroup"], [32, 32])
        self.assertEqual(run["shaders"], [12, 12, 12])
        self.assertEqual((run["passed"], run["total"], run["skipped"], run["failed_count"]), (6, 10, 3, 1))

    def test_failures_keep_their_message(self):
        self.assertEqual(rec.parse_log(EMULATED)["failed"]["cs_w64_b"], "exit 1: lane 40 differs")

    def test_skip_reasons_are_counted_most_frequent_first(self):
        self.assertEqual(list(rec.parse_log(NATIVE)["skips"].items()),
                         [("no 64-wide compute subgroup on this device", 2), ("run by d3d12_m3_frame_test", 1)])

    def test_summary_without_skipped(self):
        run = rec.parse_log("device: X (subgroup 4..8)\n3/3 passed\n")
        self.assertEqual((run["skipped"], run["failed_count"], run["shaders"]), (0, 0, None))

    def test_the_last_summary_line_counts(self):
        self.assertEqual(rec.parse_log("device: X (subgroup 4..8)\n1/2 passed\n2/2 passed\n")["passed"], 2)

    def test_other_text_is_refused(self):
        for text in ("", "hello\n", "device: X (subgroup 4..8)\n", "3/3 passed\n"):
            with self.assertRaises(rec.LogError):
                rec.parse_log(text)


class Driver(unittest.TestCase):
    def test_the_named_device(self):
        self.assertEqual(rec.driver_of("Vendor One 9000", VULKANINFO), ("VENDOR 617.42", "1.4.351"))
        self.assertEqual(rec.driver_of("Vendor Two Graphics", VULKANINFO), ("25.10.36", "1.4.315"))

    def test_unknown_device_or_no_vulkaninfo(self):
        self.assertEqual(rec.driver_of("Other", VULKANINFO), (None, None))
        self.assertEqual(rec.driver_of("Vendor One 9000", None), (None, None))


class Field(unittest.TestCase):
    def test_key_is_padded_and_long_text_wraps_under_it(self):
        text = rec.field("Tools", " ".join(["word"] * 60))
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("Tools:      word"))
        self.assertTrue(all(len(line) <= rec.WIDTH for line in lines))
        self.assertTrue(all(line.startswith(" " * 12 + "word") for line in lines[1:]))

    def test_a_long_key_gets_one_space(self):
        self.assertTrue(rec.field("run_gpu_tests.py", "x").startswith("run_gpu_tests.py: x"))


class Record(unittest.TestCase):
    def runs(self, *texts):
        return list(zip(("native", "LEONIDA_W64_EMULATE=1"), map(rec.parse_log, texts)))

    def test_both_runs(self):
        text = rec.record(self.runs(NATIVE, EMULATED), FACTS, ["Built with Ninja."])
        self.assertIn("GPU:        Vendor One 9000, subgroup 32..32\n", text)
        self.assertIn("Driver:     VENDOR 617.42, Vulkan 1.4.351\n", text)
        self.assertIn(f"Commit:     {'a' * 40} (2026-11-26)\n", text)
        self.assertIn("run_gpu_tests.py: 12 shaders: 12 translated, 12 pass spirv-val\n", text)
        self.assertIn("native:                6/10 passed, 3 skipped, 1 failed\n", text)
        self.assertIn("LEONIDA_W64_EMULATE=1: 7/10 passed, 1 skipped, 2 failed\n", text)
        self.assertIn("Failed:     cs_atomic, cs_w64_b.\n", text)
        self.assertIn("cs_w64_b (LEONIDA_W64_EMULATE=1 only): exit 1: lane 40 differs\n", text)
        self.assertIn('native: 2 "no 64-wide compute subgroup on this device", 1 "run by d3d12_m3_frame_test".', text)
        self.assertTrue(text.endswith("Notes:\n- Built with Ninja.\n"))

    def test_the_same_failures_and_skips_in_both_runs(self):
        text = rec.record(self.runs(NATIVE, NATIVE), FACTS, None)
        self.assertIn("Failed:     cs_atomic (native and emulated alike).\n", text)
        self.assertIn("Skipped:    the same in both runs: 2 ", text)
        self.assertIn("- REPLACE THIS LINE", text)

    def test_nothing_failed_or_skipped(self):
        clean = "device: X (subgroup 32..64)\n  PASS  a\n1/1 passed, 0 skipped\n"
        text = rec.record(self.runs(clean), FACTS, None)
        self.assertIn("run_gpu_tests.py: native: 1/1 passed, 0 skipped, 0 failed\n", text)
        self.assertIn("Failed:     none.\n", text)
        self.assertIn("Skipped:    none.\n", text)
        self.assertIn("Failed:     none, in either run.\n", rec.record(self.runs(clean, clean), FACTS, None))

    def test_facts_the_machine_did_not_give(self):
        text = rec.record(self.runs(NATIVE), dict(FACTS, driver=None, api=None, commit=None, date=""), None)
        self.assertIn("Driver:     (not found by vulkaninfo: add it)\n", text)
        self.assertIn("Commit:     (not a git checkout: add it)\n", text)


class Main(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def log(self, name, text):
        path = os.path.join(self.dir.name, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def test_prints_the_record(self):
        code, out, _ = run_main(["--native", self.log("n.log", NATIVE), "--emulated", self.log("e.log", EMULATED),
                                 "--note", "one", "--note", "two"])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("GPU:        Vendor One 9000"))
        self.assertTrue(out.endswith("Notes:\n- one\n- two\n"))

    def test_out_is_written_once_with_lf(self):
        target = os.path.join(self.dir.name, "record.txt")
        argv = ["--native", self.log("n.log", NATIVE), "--out", target]
        self.assertEqual(run_main(argv)[0], 0)
        with open(target, "rb") as f:
            data = f.read()
        self.assertTrue(data.startswith(b"GPU:"))
        self.assertNotIn(b"\r", data)
        code, _, err = run_main(argv)
        self.assertEqual(code, 2)
        self.assertIn("record.txt", err)
        with open(target, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_logs_it_cannot_use(self):
        native = self.log("n.log", NATIVE)
        other = self.log("o.log", EMULATED.replace("Vendor One 9000", "Vendor Two Graphics"))
        for argv, word in ((["--native", os.path.join(self.dir.name, "missing.log")], "missing.log"),
                           (["--native", self.log("bad.log", "hello\n")], "run_gpu_tests.py"),
                           (["--native", native, "--emulated", other], "different devices")):
            code, out, err = run_main(argv)
            self.assertEqual((code, out), (2, ""))
            self.assertIn(word, err)


W64G = """\
gs_w64_a_ring                A  ok
ps_w64_a_skip                A  ok
vs_w64_b_waterfall           B  ok
vs_w64_c_lane_id             C  SKIP (tier C needs W64G_REF)
vs_w64_c_readlane            C  SKIP (tier C needs W64G_REF)
"""


class W64Graphics(unittest.TestCase):
    def test_ok_per_tier_and_skip_reasons(self):
        self.assertEqual(rec.w64_note(W64G), 'w64_graphics.py: 3 ok (2 tier A, 1 tier B), 2 tier C skipped '
                                             '("tier C needs W64G_REF"), none failed.')

    def test_a_failure_is_named_with_what_it_printed(self):
        note = rec.w64_note(W64G + "ps_w64_c_x                   C  pixel 3 differs\n")
        self.assertTrue(note.endswith(", 1 failed: ps_w64_c_x (tier C): pixel 3 differs."))

    def test_other_text_is_refused(self):
        with self.assertRaises(rec.LogError):
            rec.w64_note(NATIVE)

    def test_main_puts_it_first_in_notes(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, text in (("n.log", NATIVE), ("w.log", W64G)):
                paths.append(os.path.join(d, name))
                with open(paths[-1], "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
            code, out, _ = run_main(["--native", paths[0], "--w64-graphics", paths[1], "--note", "mine"])
            self.assertEqual(code, 0)
            self.assertIn("Notes:\n- w64_graphics.py: 3 ok", out)
            self.assertTrue(out.endswith("- mine\n"))
            out = run_main(["--native", paths[0], "--w64-graphics", paths[1]])[1]
            self.assertIn("- REPLACE THIS LINE", out)
            code, _, err = run_main(["--native", paths[0], "--w64-graphics", paths[0]])
            self.assertEqual(code, 2)
            self.assertIn("w64_graphics.py", err)


class TableRow(unittest.TestCase):
    def runs(self, *texts):
        return list(zip(("native", "LEONIDA_W64_EMULATE=1"), map(rec.parse_log, texts)))

    def test_cells(self):
        row = rec.table_row(self.runs(NATIVE, EMULATED), FACTS, "Vendor.txt")
        cells = [cell.strip() for cell in row.strip().strip("|").split(" | ")]
        self.assertEqual(len(cells), 8)
        self.assertEqual(cells[0], "Vendor One 9000, subgroup 32..32")
        self.assertEqual(cells[1], "VENDOR 617.42, Vulkan 1.4.351, Windows 11, 10.0.1")
        self.assertEqual(cells[2], "`aaaaaaa`")
        self.assertEqual(cells[3], "12/12 translated, 12 valid (run_gpu_tests.py)")
        self.assertEqual(cells[5], "6/10 passed, 3 skipped, 1 failed (native); "
                                   "7/10 passed, 1 skipped, 2 failed (`LEONIDA_W64_EMULATE=1`)")
        self.assertEqual(cells[6], "`cs_atomic`, `cs_w64_b`")
        self.assertEqual(cells[7], "[local note](../gpu-test-results/Vendor.txt)")
        self.assertTrue(row.endswith(" |\n"))

    def test_equal_runs_and_missing_facts(self):
        clean = "device: X (subgroup 32..64)\n1/1 passed\n"
        row = rec.table_row(self.runs(clean, clean), dict(FACTS, driver=None, api=None, commit=None), None)
        self.assertIn("| 1/1 passed, 0 skipped, 0 failed (native and `LEONIDA_W64_EMULATE=1` alike) | none |", row)
        self.assertIn("| Windows 11, 10.0.1 | (add it) | not run | not run |", row)
        self.assertTrue(row.endswith("| (link the record) |\n"))

    def test_main_prints_it_alone_or_after_out(self):
        with tempfile.TemporaryDirectory() as d:
            log, target = os.path.join(d, "n.log"), os.path.join(d, "Mine.txt")
            with open(log, "w", encoding="utf-8", newline="\n") as f:
                f.write(NATIVE)
            code, out, _ = run_main(["--native", log, "--table-row"])
            self.assertEqual(code, 0)
            self.assertTrue(out.startswith("| Vendor One 9000") and out.count("\n") == 1)
            code, out, _ = run_main(["--native", log, "--table-row", "--out", target])
            self.assertEqual(code, 0)
            self.assertIn("[local note](../gpu-test-results/Mine.txt)", out)
            self.assertTrue(os.path.isfile(target))


class Machine(unittest.TestCase):
    def test_reads_tools_through_their_output(self):
        answers = {"vulkaninfo": VULKANINFO, "rev-parse": "b" * 40 + "\n", "log": "2026-11-25\n",
                   "llvm-mc": "LLVM (http://llvm.org/):\n  LLVM version 21.1.8\n",
                   "spirv-val": "SPIRV-Tools v2026.4 unknown hash\n"}

        def output_of(cmd):
            return next((text for key, text in answers.items() if any(key in part for part in cmd)), None)

        with mock.patch.object(rec, "output_of", output_of), \
                mock.patch.object(rec.mc_command, "mc_command", lambda raw: ["llvm-mc"]), \
                mock.patch.object(rec.shutil, "which", lambda name: name):
            facts = rec.machine("Vendor One 9000")
        self.assertEqual((facts["driver"], facts["api"]), ("VENDOR 617.42", "1.4.351"))
        self.assertEqual((facts["commit"], facts["date"]), ("b" * 40, "2026-11-25"))
        self.assertEqual(facts["tools"][:2], ["llvm-mc 21.1.8", "SPIRV-Tools v2026.4"])
        self.assertTrue(facts["tools"][2].startswith("Python "))

    def test_nothing_installed(self):
        with mock.patch.object(rec, "output_of", lambda cmd: None), \
                mock.patch.object(rec.mc_command, "mc_command", mock.Mock(side_effect=ValueError)), \
                mock.patch.object(rec.shutil, "which", lambda name: None):
            facts = rec.machine("Vendor One 9000")
        self.assertEqual((facts["driver"], facts["commit"], facts["date"]), (None, None, ""))
        self.assertEqual(len(facts["tools"]), 1)


if __name__ == "__main__":
    unittest.main()
