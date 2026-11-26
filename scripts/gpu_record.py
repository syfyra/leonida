"""Write a gpu-test-results/ record from the saved output of run_gpu_tests.py.

    python -I scripts/run_gpu_tests.py --build runtime/build > native.log
    LEONIDA_W64_EMULATE=1 python -I scripts/run_gpu_tests.py --build runtime/build > emulated.log
    python -I scripts/gpu_record.py --native native.log --emulated emulated.log [--w64-graphics w64.log]
                                 [--note TEXT ...] [--out FILE]

A record is written by hand today: the counts, the failed fixtures and the skip reasons are copied out of two
runs, and the device, driver, system, commit and tool versions are looked up one by one. This reads all of it:

- from each log: the device line with its subgroup range, how many shaders were translated and validated, the
  passed/total and skipped counts, every `FAIL` line with its message and the `SKIP` reasons with their counts;
- from the machine: the driver of that device (`vulkaninfo`), the system, the commit that is checked out, and the
  versions of llvm-mc, SPIRV-Tools and Python.

It prints the record in the fields the files in gpu-test-results/ use (`GPU:`, `Driver:`, `Subgroup:`, `OS:`,
`Commit:`, `Tools:`, `run_gpu_tests.py:`, `Failed:`, `Skipped:`, `Notes:`), or writes it to `--out`, which must
not exist yet. What a log cannot say stays yours to add: the GPU's family and PCI id, the compiler and CMake, and
in `Notes:` how it was built and whether it was AI-assisted (`--note` adds a line; without one the record ends
with a reminder line to replace). `--w64-graphics` takes the saved output of w64_graphics.py and adds a line with
its ok counts per tier, its skip reasons and what failed. `--table-row` prints the row for the table in
docs/gpu_test_results.md from the same facts (its `check.py` cell says `not run`: correct it if you ran it).

Run it on the machine and in the checkout the logs come from, right after the runs: the driver, the commit and
the tools are read now, not from the logs. It starts `vulkaninfo`, `git`, `llvm-mc --version` and
`spirv-val --version` and reads their output; it runs no test. Exit 0, or 2 for a log it cannot read or an
`--out` that exists. Standard library only; no game files.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import mc_command  # noqa: E402

DEVICE = re.compile(r"(?m)^device:\s*(.+?)\s*\(subgroup\s+(\d+)\.\.(\d+)\)\s*$")
SHADERS = re.compile(r"(?m)^(\d+) shaders: (\d+) translated, (\d+) pass spirv-val")
TOTAL = re.compile(r"(?m)^(\d+)/(\d+) passed(?:, (\d+) skipped)?\s*$")
RESULT = re.compile(r"(?m)^  (FAIL|SKIP)  (\S+)\s+(.*?)\s*$")
W64 = re.compile(r"(?m)^(\S+)\s+([ABC])\s+(\S.*?)\s*$")
WIDTH = 112
REMINDER = "REPLACE THIS LINE: how it was built (generator, compiler), what else ran, and whether it was AI-assisted."


class LogError(Exception):
    pass


def parse_log(text):
    """What one run printed: device, subgroup, shader counts, passed/total/skipped, failures and skip reasons."""
    device, total = DEVICE.search(text), None
    for total in TOTAL.finditer(text):
        pass                                    # the summary is the last such line
    if not device or not total:
        raise LogError("no `device:` line and `N/M passed` summary: is this the output of run_gpu_tests.py?")
    shaders = SHADERS.search(text)
    failed, skips = {}, Counter()
    for kind, name, message in RESULT.findall(text):
        if kind == "FAIL":
            failed.setdefault(name, message)
        else:
            skips[re.sub(r"^SKIP\s*\((.*)\)$", r"\1", message)] += 1
    passed, count, skipped = int(total.group(1)), int(total.group(2)), int(total.group(3) or 0)
    return {"device": device.group(1), "subgroup": [int(device.group(2)), int(device.group(3))],
            "shaders": [int(g) for g in shaders.groups()] if shaders else None,
            "passed": passed, "total": count, "skipped": skipped, "failed_count": count - passed - skipped,
            "failed": failed, "skips": dict(skips.most_common())}


def w64_note(text):
    """A Notes: line for the output of w64_graphics.py: ok per tier, the skip reasons, and what failed."""
    rows = W64.findall(text)
    if not rows:
        raise LogError("no `name  A|B|C  result` line: is this the output of w64_graphics.py?")
    ok, skips, failed = Counter(), Counter(), []
    for name, tier, result in rows:
        if result == "ok":
            ok[tier] += 1
        elif result.startswith("SKIP"):
            skips[(tier, re.sub(r"^SKIP\s*\((.*)\)$", r"\1", result))] += 1
        else:
            failed.append(f"{name} (tier {tier}): {result}")
    tiers = ", ".join(f"{ok[t]} tier {t}" for t in "ABC" if ok[t])
    note = f"w64_graphics.py: {sum(ok.values())} ok" + (f" ({tiers})" if tiers else "")
    for (tier, reason), n in skips.most_common():
        note += f', {n} tier {tier} skipped ("{reason}")'
    return note + (f", {len(failed)} failed: " + "; ".join(failed) if failed else ", none failed") + "."


def output_of(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (r.stdout + r.stderr).decode("utf-8", "replace") if r.returncode == 0 else None


def driver_of(device, text):
    """`driverInfo` and `apiVersion` of the device called `device` in `vulkaninfo --summary` text."""
    for block in re.split(r"(?m)^GPU\d+:\s*$", text or "")[1:]:
        name = re.search(r"(?m)^\s*deviceName\s*=\s*(.+?)\s*$", block)
        if name and name.group(1) == device:
            info = re.search(r"(?m)^\s*driverInfo\s*=\s*(.+?)\s*$", block)
            name_ = re.search(r"(?m)^\s*driverName\s*=\s*(.+?)\s*$", block)
            api = re.search(r"(?m)^\s*apiVersion\s*=\s*(\S+)", block)
            parts = [p.group(1) for p in (name_, info) if p]
            return " ".join(parts) or None, api.group(1) if api else None
    return None, None


def machine(device, root=ROOT, llvm_mc=None):
    """Driver, system, commit and tools as they are now."""
    driver, api = driver_of(device, output_of(["vulkaninfo", "--summary"]))
    commit = (output_of(["git", "-C", root, "rev-parse", "HEAD"]) or "").strip() or None
    date = (output_of(["git", "-C", root, "log", "-1", "--format=%cs", "HEAD"]) or "").strip() if commit else ""
    tools = []
    try:
        mc = output_of([*mc_command.mc_command(llvm_mc), "--version"])
    except (ValueError, OSError):
        mc = None
    version = re.search(r"LLVM version (\S+)", mc or "")
    if version:
        tools.append(f"llvm-mc {version.group(1)}")
    val = shutil.which("spirv-val")
    spirv = re.search(r"SPIRV-Tools (v\S+)", output_of([val, "--version"]) or "") if val else None
    if spirv:
        tools.append(f"SPIRV-Tools {spirv.group(1)}")
    tools.append(f"Python {platform.python_version()}")
    system = f"{platform.system()} {platform.release()}, {platform.version()}".strip(", ")
    return {"driver": driver, "api": api, "os": system, "commit": commit, "date": date, "tools": tools}


def field(key, text):
    """One field: the key padded to twelve columns, the text wrapped with its continuation lines indented."""
    lines, pad = [], " " * 12
    for part in text.split("\n"):
        line = ""
        for word in part.split(" "):
            if line and len(pad) + len(line) + 1 + len(word) > WIDTH:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        lines.append(line)
    head = f"{key + ':':<12}" if len(key) < 11 else f"{key}: "
    return head + ("\n" + pad).join(lines) + "\n"


def counts(run):
    return f"{run['passed']}/{run['total']} passed, {run['skipped']} skipped, {run['failed_count']} failed"


def failed_text(runs):
    """`none`, or the fixtures first and then a line of its own for each with what its run printed."""
    names = sorted({name for _, run in runs for name in run["failed"]})
    if not names:
        return "none" + (", in either run." if len(runs) > 1 else ".")
    where = {name: [label for label, run in runs if name in run["failed"]] for name in names}
    same = len(runs) > 1 and all(len(labels) == len(runs) for labels in where.values())
    head = ", ".join(names) + (" (native and emulated alike)." if same else ".")
    lines = [head]
    for name in names:
        message = next(run["failed"][name] for _, run in runs if name in run["failed"])
        only = "" if same or len(runs) == 1 else f" ({' and '.join(where[name])} only)" if len(where[name]) == 1 else ""
        lines.append(f"{name}{only}: {message}")
    return "\n".join(lines)


def skipped_text(runs):
    parts = []
    for label, run in runs:
        reasons = ", ".join(f'{n} "{reason}"' for reason, n in run["skips"].items()) or "none"
        parts.append((f"{label}: " if len(runs) > 1 else "") + reasons + ".")
    if len(runs) > 1 and runs[0][1]["skips"] == runs[1][1]["skips"]:
        return "the same in both runs: " + parts[0].split(": ", 1)[1]
    return "\n".join(parts)


def record(runs, facts, notes):
    """The record text for [(label, run)], the machine's facts and the note lines."""
    first = runs[0][1]
    low, high = first["subgroup"]
    out = field("GPU", f"{first['device']}, subgroup {low}..{high}")
    out += field("Driver", (facts["driver"] or "(not found by vulkaninfo: add it)")
                 + (f", Vulkan {facts['api']}" if facts["api"] else ""))
    out += field("Subgroup", f"{low}..{high} (the run_gpu_tests.py device line)")
    out += field("OS", facts["os"])
    out += field("Commit", (facts["commit"] or "(not a git checkout: add it)")
                 + (f" ({facts['date']})" if facts["date"] else ""))
    out += field("Tools", ", ".join(facts["tools"]))
    shaders = first["shaders"]
    head = f"{shaders[0]} shaders: {shaders[1]} translated, {shaders[2]} pass spirv-val" if shaders else ""
    if len(runs) == 1 and not head:
        out += field("run_gpu_tests.py", f"{runs[0][0]}: {counts(first)}")
    else:
        width = max(len(label) for label, _ in runs) + 2
        out += field("run_gpu_tests.py", "\n".join(
            ([head] if head else []) + [f"{label + ':':<{width}}{counts(run)}" for label, run in runs]))
    out += field("Failed", failed_text(runs))
    out += field("Skipped", skipped_text(runs))
    out += "Notes:\n" + "".join(f"- {note}\n" for note in notes or [REMINDER])
    return out


def table_row(runs, facts, entry):
    """The row of the table in docs/gpu_test_results.md; `entry` is the record's file name in gpu-test-results/."""
    first = runs[0][1]
    low, high = first["subgroup"]
    shaders = first["shaders"]
    names = sorted({name for _, run in runs for name in run["failed"]})
    same = len(runs) > 1 and counts(runs[0][1]) == counts(runs[1][1])
    cells = [
        f"{first['device']}, subgroup {low}..{high}",
        ", ".join(part for part in (facts["driver"], f"Vulkan {facts['api']}" if facts["api"] else "",
                                    facts["os"]) if part),
        f"`{facts['commit'][:7]}`" if facts["commit"] else "(add it)",
        f"{shaders[1]}/{shaders[0]} translated, {shaders[2]} valid (run_gpu_tests.py)" if shaders else "not run",
        "not run",
        f"{counts(first)} (native and `LEONIDA_W64_EMULATE=1` alike)" if same else
        "; ".join(f"{counts(run)} ({'native' if label == 'native' else f'`{label}`'})" for label, run in runs),
        ", ".join(f"`{name}`" for name in names) or "none",
        f"[local note](../gpu-test-results/{entry})" if entry else "(link the record)",
    ]
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--native", required=True, metavar="LOG", help="saved output of run_gpu_tests.py")
    ap.add_argument("--emulated", metavar="LOG", help="the same with LEONIDA_W64_EMULATE=1")
    ap.add_argument("--w64-graphics", metavar="LOG", help="saved output of w64_graphics.py, summed up in Notes:")
    ap.add_argument("--note", action="append", metavar="TEXT", help="a line for Notes: (repeat for more)")
    ap.add_argument("--table-row", action="store_true",
                    help="print the row for docs/gpu_test_results.md: instead of the record, or after --out wrote it")
    ap.add_argument("--llvm-mc", help="llvm-mc path or whole command, when it is not in LLVM_MC or on PATH")
    ap.add_argument("--out", metavar="FILE", help="write the record here instead of printing it; must not exist")
    args = ap.parse_args(argv)
    runs = []
    try:
        for label, path in (("native", args.native), ("LEONIDA_W64_EMULATE=1", args.emulated)):
            if path:
                with open(path, encoding="utf-8", errors="replace") as f:
                    runs.append((label, parse_log(f.read())))
        if len(runs) == 2 and runs[0][1]["device"] != runs[1][1]["device"]:
            raise LogError(f"the two logs are of different devices: {runs[0][1]['device']} and {runs[1][1]['device']}")
        notes = list(args.note or [REMINDER])
        if args.w64_graphics:
            with open(args.w64_graphics, encoding="utf-8", errors="replace") as f:
                notes.insert(0, w64_note(f.read()))
    except (OSError, LogError) as e:
        print(f"gpu_record: {e}", file=sys.stderr)
        return 2
    facts = machine(runs[0][1]["device"], llvm_mc=args.llvm_mc)
    text = record(runs, facts, notes)
    if not args.out:
        sys.stdout.write(table_row(runs, facts, None) if args.table_row else text)
        return 0
    try:
        with open(args.out, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except OSError as e:
        print(f"gpu_record: {args.out}: {e.strerror or e}", file=sys.stderr)
        return 2
    print(f"wrote {args.out}")
    if args.table_row:
        sys.stdout.write(table_row(runs, facts, os.path.basename(args.out)))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
