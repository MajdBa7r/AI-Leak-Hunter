"""One-command check + launch for a laptop or conference booth (Windows, macOS, Linux).

    python scripts/booth_check.py              # smoke tests, then launch the app
    python scripts/booth_check.py --offline    # same, copilot forced to offline mode
    python scripts/booth_check.py --full       # full test suite (~3-4 min) before launch
    python scripts/booth_check.py --no-launch  # tests only

Run from the repository root after:  pip install -r requirements-dev.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPILOT_VARS = {"OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LEAK_HUNTER_LLM_PROVIDER",
                "OPENAI_MODEL", "GROQ_MODEL", "GEMINI_MODEL", "LEAK_HUNTER_COPILOT_OFFLINE"}
SMOKE = ["tests/test_copilot.py", "tests/test_product.py", "tests/test_exposure.py"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="force the offline copilot (no LLM calls)")
    ap.add_argument("--full", action="store_true", help="run the full test suite instead of the smoke tests")
    ap.add_argument("--no-launch", action="store_true", help="run the checks only")
    ap.add_argument("--port", default="8501")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    # Tests control provider settings themselves: never pass real keys or the offline switch to them.
    test_env = {k: v for k, v in env.items() if k not in COPILOT_VARS}
    app_env = dict(env)
    if args.offline:
        app_env["LEAK_HUNTER_COPILOT_OFFLINE"] = "true"

    print(">> Building detection outputs (outputs/) ...", flush=True)
    rc = subprocess.call([sys.executable, "-m", "leak_hunter.pipeline", "--out-dir", "outputs"], cwd=ROOT, env=test_env)
    if rc:
        print("!! Detection pipeline failed - see the error above.")
        return rc

    tests = [] if args.full else SMOKE
    print(f">> Running {'full' if args.full else 'smoke'} tests ...", flush=True)
    rc = subprocess.call([sys.executable, "-m", "pytest", *tests], cwd=ROOT, env=test_env)
    if rc:
        print("!! Tests failed - do not demo until this is fixed.")
        return rc
    print(">> All checks passed.")

    if args.no_launch:
        return 0
    mode = "OFFLINE copilot" if args.offline else "copilot uses an API key if one is set"
    print(f">> Launching http://localhost:{args.port}  ({mode})  - Ctrl+C to stop", flush=True)
    return subprocess.call([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                            "--server.port", args.port], cwd=ROOT, env=app_env)


if __name__ == "__main__":
    sys.exit(main())
