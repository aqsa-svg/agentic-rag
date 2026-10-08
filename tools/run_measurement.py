"""Run a measurement script so that a FAILURE cannot look like a SUCCESS.

The hole this closes, found 2026-10-08. Measurements were launched as

    python script.py > out 2>&1; echo "exit $?"

and the trailing `echo` always succeeds, so the shell reported exit 0 even when the script
raised. A syntax error I had introduced into a v4 re-run then looked like a completed run
that computed nothing - no RESULT lines, exit 0, "it ran." That is the same shape as the
instrument failures in docs/SILENT_WRONGNESS.md instance 4: a probe that stringified a
`(text, stats)` tuple and reported "0 missing", and an nDCG>0 check that printed "EARNS IT"
while measuring the wrong thing. By that point the instruments had been wrong more often
than the system they measured.

This runner removes the hole for every future measurement:

* it **compiles the script first**, so a syntax error is caught before anything appears to
  run, and reported as such;
* it runs the script and **exits with the script's own status**, so a non-zero exit
  propagates instead of being masked by a trailing shell command;
* on failure it writes a loud, greppable line to stderr, so a swallowed traceback cannot
  pass for a clean run.

Usage:  python tools/run_measurement.py path/to/measurement.py [args...]
A measurement is TRUSTED only when this exits 0 AND prints the lines it was meant to.
"""

from __future__ import annotations

import py_compile
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_measurement.py <script.py> [args...]", file=sys.stderr)
        return 2

    script = sys.argv[1]
    try:
        py_compile.compile(script, doraise=True)
    except py_compile.PyCompileError as exc:
        print(f"MEASUREMENT NOT RUN - {script} did not compile:\n{exc}", file=sys.stderr)
        return 2

    result = subprocess.run([sys.executable, script, *sys.argv[2:]])
    if result.returncode != 0:
        print(
            f"MEASUREMENT FAILED - {script} exited {result.returncode}; its output is NOT a "
            "result and must not be read as one.",
            file=sys.stderr,
        )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
