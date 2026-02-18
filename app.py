import os
import subprocess
import sys


def launch_streamlit() -> None:
    app_path = os.path.abspath("UI/main.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    env = os.environ.copy()
    subprocess.Popen(cmd, env=env).wait()

if __name__ == "__main__":
    launch_streamlit()
