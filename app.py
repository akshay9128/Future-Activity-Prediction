"""
Application Entrypoint for the Interactive Web Dashboard.

Run directly with:
    streamlit run app.py
or:
    python app.py
"""

from pathlib import Path
import subprocess
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.streamlit_app import main

if __name__ == "__main__":
    # If executed with standard python rather than streamlit run, launch streamlit subprocess
    if "streamlit" not in sys.argv[0]:
        streamlit_script = PROJECT_ROOT / "src" / "app" / "streamlit_app.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_script)]
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\nApplication stopped by user.")
    else:
        main()
