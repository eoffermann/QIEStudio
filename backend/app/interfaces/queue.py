"""Job queue seam (DESIGN §4.1a / §4.1).

v1 is an in-process **ordered** queue with a single worker thread, guaranteeing **one
inference at a time** per accelerator (DESIGN §9.3). The interface is broker-agnostic so a
Redis/RQ/Celery implementation can replace it for multi-worker/cloud without touching request
handlers, which stay stateless.

Beyond FIFO submit, the queue supports **management of pending jobs** (DESIGN §13 queue-
management): list the pending order, remove a queued job, and reorder/prioritize the pending
jobs. Cancellation is cooperative: a running task periodically checks
:meth:`JobQueue.is_canceled` (the pipeline step-callback does this) and aborts cleanly.
Job *state* lives in the DB (status persisted), so jobs survive process restarts; the job
service re-enqueues any ``queued``/``running`` rows on startup.
"""

from __future__ import annotations

import logging
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
    def pending_ids(self) -> list[str]:
        """The job ids currently waiting to run, in execution order (excludes the running one)."""

    @abstractmethod
    def remove_pending(self, job_id: str) -> bool:
        """Remove a not-yet-started job from the queue. Returns True if it was pending."""

    @abstractmethod
    def reorder_pending(self, ordered_ids: list[str]) -> None:
        """Reorder pending jobs to match ``ordered_ids`` (unlisted ids keep relative order)."""

    @abstractmethod
    def start(self) -> None:
        """Start the worker (idempotent)."""

    @abstractmethod
    def shutdown(self, *, wait: bool = True) -> None:
        """Stop the worker, optionally draining the in-flight task."""


@dataclass
class InProcessJobQueue(JobQueue):
    """Single background worker thread executing tasks one at a time, in a mutable order."""

    _pending: list[QueuedTask] = field(default_factory=list)
    _canceled: set[str] = field(default_factory=set)
    _cond: threading.Condition = field(default_factory=threading.Condition)
    _worker: threading.Thread | None = None
    _running_job: str | None = None
    _stop: bool = False

    def submit(self, task: QueuedTask) -> None:
        log.info("Enqueue job %s (%s)", task.job_id, task.label or "job")
        with self._cond:
            self._pending.append(task)
            self._cond.notify()

    def request_cancel(self, job_id: str) -> None:
        with self._cond:
            self._canceled.add(job_id)
        log.info("Cancellation requested for job %s", job_id)

    def is_canceled(self, job_id: str) -> bool:
        with self._cond:
            return job_id in self._canceled

    def pending_ids(self) -> list[str]:
        with self._cond:
            return [t.job_id for t in self._pending]

    def remove_pending(self, job_id: str) -> bool:
        with self._cond:
            for i, task in enumerate(self._pending):
                if task.job_id == job_id:
                    del self._pending[i]
                    return True
            return False

    def reorder_pending(self, ordered_ids: list[str]) -> None:
        with self._cond:
            rank = {jid: i for i, jid in enumerate(ordered_ids)}
            # Listed ids first (in the requested order), then any others keeping their order.
            self._pending.sort(key=lambda t: (rank.get(t.job_id, len(rank)),))
        log.info("Reordered pending queue to %s", ordered_ids)

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
        while True:
            with self._cond:
                while not self._pending and not self._stop:
                    self._cond.wait()
                if self._stop:
                    break
                task = self._pending.pop(0)
                self._running_job = task.job_id
            try:
                if self.is_canceled(task.job_id):
                    log.info("Job %s canceled before start; skipping", task.job_id)
                else:
                    task.fn()
            except Exception:  # noqa: BLE001 — worker must never die on a task error
                log.exception("Job %s failed in worker", task.job_id)
            finally:
                with self._cond:
                    self._running_job = None
                    self._canceled.discard(task.job_id)

    def shutdown(self, *, wait: bool = True) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
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
