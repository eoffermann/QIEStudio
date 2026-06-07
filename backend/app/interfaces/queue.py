"""Job queue seam (DESIGN §4.1a / §4.1).

v1 is an in-process FIFO with a single worker thread, guaranteeing **one inference at a
time** per accelerator (DESIGN §9.3). The interface is broker-agnostic so a Redis/RQ/Celery
implementation can replace it for multi-worker/cloud without touching request handlers,
which stay stateless.

Cancellation is cooperative: a running task periodically checks
:meth:`JobQueue.is_canceled` (the pipeline step-callback does this) and aborts cleanly.
Job *state* lives in the DB (status persisted), so jobs survive process restarts; the job
service re-enqueues any ``queued``/``running`` rows on startup.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# A task is a no-arg callable that performs the work for one job id. It should consult the
# queue's cancellation flag cooperatively and persist its own progress/status to the DB.
TaskFn = Callable[[], None]


@dataclass
class QueuedTask:
    job_id: str
    fn: TaskFn
    label: str = ""


class JobQueue(ABC):
    """Abstract single-accelerator job queue."""

    @abstractmethod
    def submit(self, task: QueuedTask) -> None:
        """Enqueue a task for serialized execution."""

    @abstractmethod
    def request_cancel(self, job_id: str) -> None:
        """Request cooperative cancellation of a queued or running job."""

    @abstractmethod
    def is_canceled(self, job_id: str) -> bool:
        """Whether cancellation has been requested for this job."""

    @abstractmethod
    def start(self) -> None:
        """Start the worker (idempotent)."""

    @abstractmethod
    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the worker, optionally draining the in-flight task."""


@dataclass
class InProcessJobQueue(JobQueue):
    """Single background worker thread executing tasks FIFO, one at a time."""

    _q: queue.Queue[QueuedTask | None] = field(default_factory=queue.Queue)
    _canceled: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _worker: threading.Thread | None = None
    _running_job: str | None = None
    _stop: bool = False

    def submit(self, task: QueuedTask) -> None:
        log.info("Enqueue job %s (%s)", task.job_id, task.label or "job")
        self._q.put(task)

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            self._canceled.add(job_id)
        log.info("Cancellation requested for job %s", job_id)

    def is_canceled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._canceled

    @property
    def running_job(self) -> str | None:
        return self._running_job

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop = False
        self._worker = threading.Thread(target=self._run, name="qie-job-worker", daemon=True)
        self._worker.start()
        log.info("In-process job worker started")

    def _run(self) -> None:
        while not self._stop:
            task = self._q.get()
            if task is None:  # shutdown sentinel
                self._q.task_done()
                break
            self._running_job = task.job_id
            try:
                if self.is_canceled(task.job_id):
                    log.info("Job %s canceled before start; skipping", task.job_id)
                else:
                    task.fn()
            except Exception:  # noqa: BLE001 — worker must never die on a task error
                log.exception("Job %s failed in worker", task.job_id)
            finally:
                self._running_job = None
                with self._lock:
                    self._canceled.discard(task.job_id)
                self._q.task_done()

    def shutdown(self, *, wait: bool = True) -> None:
        self._stop = True
        self._q.put(None)
        if wait and self._worker:
            self._worker.join(timeout=5.0)
        log.info("In-process job worker stopped")


_default_queue: InProcessJobQueue | None = None


def get_job_queue() -> InProcessJobQueue:
    """Return the process-wide job queue (created + started lazily)."""
    global _default_queue
    if _default_queue is None:
        _default_queue = InProcessJobQueue()
        _default_queue.start()
    return _default_queue
