"""Release distillation workers and remove only this run's teacher targets."""

import fcntl
import logging
import os
from pathlib import Path
import shutil
import signal
import threading
import weakref


log = logging.getLogger(__name__)


class TeacherCacheLifecycle:
    def __init__(self, *, cleanup_on_exit=True):
        self.cleanup_on_exit = cleanup_on_exit
        self._pid = os.getpid()
        self._leases = {}
        self._caches = weakref.WeakSet()
        self._iterators = weakref.WeakSet()
        self._handlers = {}

    def acquire(self, directory):
        """Hold a shared lease before creating/opening a cache, on every rank."""
        # Do not follow a symlink in place of the fingerprint directory when
        # choosing what to delete. The workspace already resolves its parent.
        directory = Path(directory).absolute()
        if directory in self._leases:
            return
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Keep this tiny lock outside the deleted directory. Unlinking it would
        # let a new job lock a different inode while cleanup is still running.
        lease = (directory.parent / f".{directory.name}.usage.lock").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_SH)
        except BaseException:
            lease.close()
            raise
        self._leases[directory] = lease

    def track_cache(self, cache):
        self._caches.add(cache)

    def track_iterator(self, iterator):
        if hasattr(iterator, "_shutdown_workers"):
            self._iterators.add(iterator)
        return iterator

    def close_readers(self):
        # Forked DataLoader workers must never release the parent's leases.
        if os.getpid() != self._pid:
            return
        for resource in list(self._iterators):
            try:
                resource._shutdown_workers()
            except Exception:
                log.exception("Could not shut down teacher cache DataLoader workers")
        self._iterators.clear()
        for cache in list(self._caches):
            try:
                cache.close()
            except Exception:
                log.exception("Could not close teacher cache: %s", cache.directory)
        self._caches.clear()

    def close(self):
        if os.getpid() != self._pid:
            return
        self.close_readers()
        for directory, lease in list(self._leases.items()):
            try:
                fcntl.flock(lease, fcntl.LOCK_UN)
                if not self.cleanup_on_exit:
                    log.info("Keeping teacher cache (cleanup_on_exit=false): %s", directory)
                    continue
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    log.info("Teacher cache still in use; cleanup deferred to the last reader: %s", directory)
                    continue
                if directory.exists():
                    shutil.rmtree(directory)
                    log.info("Deleted teacher cache on exit: %s", directory)
            except OSError:
                # Cleanup errors must not hide the original training exception.
                log.exception("Could not delete teacher cache on exit: %s", directory)
            finally:
                lease.close()
                self._leases.pop(directory, None)

    def _handle_signal(self, signum, frame):
        if os.getpid() == self._pid:
            log.warning("Received %s; exiting and cleaning up teacher cache", signal.Signals(signum).name)
        raise SystemExit(128 + signum)

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                previous = signal.getsignal(signum)
                # Preserve nohup's explicit request to ignore hangups.
                if signum == signal.SIGHUP and previous == signal.SIG_IGN:
                    continue
                self._handlers[signum] = previous
                signal.signal(signum, self._handle_signal)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.getpid() != self._pid:
            return
        try:
            # A second Ctrl+C/stop request must not interrupt file removal.
            for signum in self._handlers:
                signal.signal(signum, signal.SIG_IGN)
            self.close()
        finally:
            for signum, previous in self._handlers.items():
                signal.signal(signum, previous)
            self._handlers.clear()
