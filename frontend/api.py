"""
frontend/api.py - The dashboard's only link to the backend: plain HTTP calls to the FastAPI server.

The frontend never imports backend code; if the API runs on another machine, only API_URL changes.
"""

import os
import time

import requests

API_URL = os.getenv("VERITY_API_URL", "http://localhost:8000")


class ApiError(RuntimeError):
    pass


def _check(response):
    if response.ok:
        return response
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise ApiError(str(detail))


def get(path, **params):
    try:
        return _check(requests.get(f"{API_URL}{path}", params=params, timeout=60)).json()
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the VerityAi API at {API_URL}. Start it with:  python start.py")


def get_bytes(path):
    return _check(requests.get(f"{API_URL}{path}", timeout=120)).content


def post(path, json=None, files=None, data=None, timeout=300):
    try:
        return _check(requests.post(f"{API_URL}{path}", json=json, files=files, data=data, timeout=timeout)).json()
    except requests.ConnectionError:
        raise ApiError(f"Cannot reach the VerityAi API at {API_URL}. Start it with:  python start.py")


def put(path, json, **params):
    return _check(requests.put(f"{API_URL}{path}", json=json, params=params, timeout=60)).json()


def delete(path, json=None):
    return _check(requests.delete(f"{API_URL}{path}", json=json, timeout=60)).json()


def wait_for_job(job_id, on_progress=None, poll=0.7):
    """Poll a background job until it finishes. on_progress(job) gets every update."""
    while True:
        job = get(f"/jobs/{job_id}")
        if on_progress:
            on_progress(job)
        if job["status"] == "done":
            return job["result"]
        if job["status"] == "error":
            raise ApiError(job["error"])
        time.sleep(poll)
