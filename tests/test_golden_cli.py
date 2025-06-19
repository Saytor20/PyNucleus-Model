import subprocess, sys
def test_eval_golden():
    proc = subprocess.run([sys.executable, "-m", "pynucleus.cli", "eval_golden"])
    assert proc.returncode == 0 