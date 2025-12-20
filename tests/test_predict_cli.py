import subprocess
import sys
from pathlib import Path

def test_predict_cli_runs():
    repo = Path(__file__).resolve().parents[1]
    test_csv = repo / "data" / "processed" / "test.csv"
    if not test_csv.exists():
        # If data isn't tracked (likely), skip
        return

    cmd = [sys.executable, "predict.py", str(test_csv)]
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    assert p.returncode == 0
