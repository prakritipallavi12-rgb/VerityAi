"""
start.py - Start VerityAi: the FastAPI backend and the Streamlit dashboard together.

    python start.py

  Backend  (API)       http://localhost:8000      (endpoint list: http://localhost:8000/docs)
  Frontend (dashboard) http://localhost:8501

Press Ctrl+C to stop both.
"""

import subprocess
import sys
import time
import webbrowser

import requests

API_PORT, UI_PORT = 8000, 8501


def main():
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.server:app", "--port", str(API_PORT)])
    # Wait for the API before opening the dashboard, so the first page load works
    for _ in range(120):
        try:
            if requests.get(f"http://localhost:{API_PORT}/health", timeout=1).ok:
                break
        except requests.RequestException:
            time.sleep(0.5)
    print(f"\n  VerityAi API running:  http://localhost:{API_PORT}/docs")
    # headless = no first-run email prompt (it would wait for typing forever); we open the browser ourselves
    ui = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(UI_PORT),
                           "--server.headless", "true", "--browser.gatherUsageStats", "false"])
    print(f"  VerityAi dashboard:    http://localhost:{UI_PORT}\n")
    time.sleep(3)
    webbrowser.open(f"http://localhost:{UI_PORT}")
    try:
        while api.poll() is None and ui.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (ui, api):
            process.terminate()
        print("\n  VerityAi stopped.")


if __name__ == "__main__":
    main()
