import os
import subprocess
import sys
import time
import webbrowser

from UI import run


def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)


def launch_streamlit() -> None:
    app_path = resource_path("app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.0)
    webbrowser.open("http://localhost:8501")
    p.wait()


def running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx(True) is not None


if __name__ == "__main__":
    if running_in_streamlit():
        run()
    else:
        launch_streamlit()
