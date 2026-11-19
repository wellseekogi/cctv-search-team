"""Persist reproducible checks and individual assertion results for handoff."""

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({"case": test.id(), "expected": "all assertions pass",
                             "actual": "all assertions passed", "status": "PASS", "verifier": "automated unittest"})


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    checks = list(result.records)
    # CLI is exercised in independent processes below; avoid mocking filesystem behavior.
    scratch = ROOT / "runs"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cctv-a-check-", dir=scratch) as tmp:
        if not Path(tmp).resolve().is_relative_to(scratch.resolve()):
            raise RuntimeError("Temporary test directory is outside runs")
        folders = [Path(tmp) / "first", Path(tmp) / "second"]
        for folder in folders:
            completed = subprocess.run([sys.executable, "-m", "sampling", "--input", "fixtures/event_trace",
                                        "--out", str(folder), "--run-id", "repro-check"], cwd=ROOT,
                                       capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr)
        actual = json.loads((folders[0] / "selection_summary.json").read_text(encoding="utf-8"))["counts"]
        expected = {"events": 9, "object": 5, "scene": 8}
        checks.append({"case": "A8 CLI counts", "expected": expected, "actual": actual,
                       "status": "PASS" if actual == expected else "FAIL", "verifier": "automated CLI"})
        hashes = {}
        for filename in ("events.jsonl", "samples.jsonl", "selection_summary.json"):
            digests = [hashlib.sha256((folder / filename).read_bytes()).hexdigest() for folder in folders]
            hashes[filename] = digests
        equal = all(pair[0] == pair[1] for pair in hashes.values())
        checks.append({"case": "two CLI runs byte equality", "expected": True, "actual": equal,
                       "sha256": hashes, "status": "PASS" if equal else "FAIL", "verifier": "automated SHA256"})
        again = subprocess.run([sys.executable, "-m", "sampling", "--out", str(folders[0])],
                               cwd=ROOT, capture_output=True, text=True)
        checks.append({"case": "nonempty output rejected", "expected": 2, "actual": again.returncode,
                       "status": "PASS" if again.returncode == 2 else "FAIL", "verifier": "automated CLI"})
    for test, trace in result.failures + result.errors:
        checks.append({"case": test.id(), "status": "FAIL", "actual": trace, "expected": "all assertions pass"})
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "command": "python tests/run_checks.py", "input": "fixtures/event_trace",
              "unit_tests": result.testsRun, "checks": checks,
              "team_review": {"F": "pending", "C": "pending", "G_Python_3_11": "pending"},
              "status": "PASS" if result.wasSuccessful() and all(c["status"] == "PASS" for c in checks) else "FAIL"}
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
