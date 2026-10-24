"""
api/jobs.py - Long tasks (generating tests, evaluating, stability checks) run as background jobs.

The API answers immediately with a job ID; the dashboard then asks GET /jobs/{id} every second
to show "Evaluating 4/10..." until the job is done.
"""

import threading
import time
import traceback
import uuid

_jobs = {}
_lock = threading.Lock()


def start(kind, work):
    """work(progress) runs in a thread; progress(done, total, message='') updates the job."""
    job_id = uuid.uuid4().hex[:10]
    job = {"job_id": job_id, "kind": kind, "status": "running", "done": 0, "total": 0, "message": "Starting…",
           "result": None, "error": None, "started": time.time(), "finished": None}
    with _lock:
        _jobs[job_id] = job

    def progress(done, total, message=""):
        job.update(done=done, total=total, message=message or f"{done}/{total}")

    def run():
        try:
            job["result"] = work(progress)
            job["status"] = "done"
        except Exception as e:
            traceback.print_exc()
            job["error"] = str(e)
            job["status"] = "error"
        job["finished"] = time.time()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return None
    elapsed = (job["finished"] or time.time()) - job["started"]
    return {**job, "elapsed": round(elapsed, 1)}
