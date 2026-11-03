"""
patch_env.py - Environment compatibility patch for POSIX SemLock in restricted sandboxes.
Replaces multiprocessing.pool.ThreadPool with ThreadPoolExecutor to prevent Errno 13 Permission denied.
"""
import multiprocessing.pool
from concurrent.futures import ThreadPoolExecutor


class SafeThreadPool:
    def __init__(self, processes=None, *args, **kwargs):
        self.executor = ThreadPoolExecutor(max_workers=max(1, processes or 2))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=False)

    def imap(self, func, iterable, chunksize=1):
        return self.executor.map(func, iterable)

    def imap_unordered(self, func, iterable, chunksize=1):
        return self.executor.map(func, iterable)

    def map(self, func, iterable, chunksize=None):
        return list(self.executor.map(func, iterable))


def apply_patch():
    multiprocessing.pool.ThreadPool = SafeThreadPool
    for mod_name in [
        "ultralytics.data.base",
        "ultralytics.data.dataset",
        "ultralytics.data.loaders",
        "ultralytics.data.converter",
        "ultralytics.utils.downloads",
        "ultralytics.engine.predictor",
    ]:
        try:
            m = __import__(mod_name, fromlist=["ThreadPool"])
            if hasattr(m, "ThreadPool"):
                setattr(m, "ThreadPool", SafeThreadPool)
        except Exception:
            pass


apply_patch()
