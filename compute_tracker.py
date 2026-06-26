import os
import threading
import time

import psutil


class ComputeTracker:
    """Quietly tracks CPU time for the current process and worker children."""

    def __init__(self, interval_seconds=1.0):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.logical_cpus = psutil.cpu_count(logical=True) or 1
        self.started_at = None
        self.finished_at = None
        self.cpu_seconds = 0.0
        self._last_sample_at = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self.started_at = time.perf_counter()
        self._last_sample_at = self.started_at
        self._prime()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self.started_at is None:
            return
        self._sample_once()
        self.finished_at = time.perf_counter()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def print_summary(self):
        end = self.finished_at or time.perf_counter()
        wall_seconds = max(0.0, end - (self.started_at or end))
        avg_cores = self.cpu_seconds / wall_seconds if wall_seconds else 0.0
        machine_percent = (avg_cores / self.logical_cpus) * 100.0

        print("\nCompute used")
        print(f"Wall time: {wall_seconds / 60.0:.1f} min")
        print(
            f"Average CPU: {avg_cores:.2f} cores "
            f"({machine_percent:.1f}% of {self.logical_cpus} cores)"
        )
        print(
            f"CPU burn: {self.cpu_seconds / 60.0:.1f} core-min "
            f"({self.cpu_seconds / 3600.0:.2f} core-hours)"
        )

    def _run(self):
        while not self._stop_event.wait(self.interval_seconds):
            self._sample_once()

    def _prime(self):
        for proc in self._tracked_processes():
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    def _sample_once(self):
        now = time.perf_counter()
        elapsed = max(0.0, now - (self._last_sample_at or now))
        percent = 0.0

        for proc in self._tracked_processes():
            try:
                percent += proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self.cpu_seconds += elapsed * (percent / 100.0)
        self._last_sample_at = now

    def _tracked_processes(self):
        try:
            return [self.process] + self.process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []
