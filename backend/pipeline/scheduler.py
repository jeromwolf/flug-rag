"""Document sync scheduler with APScheduler and optional watchdog monitoring."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings

logger = logging.getLogger(__name__)

# Optional watchdog imports
try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None
    FileSystemEvent = None
    logger.warning("watchdog not available - file system monitoring disabled")


class SyncScheduler:
    """Automated document synchronization scheduler.

    Features:
    - Cron-based scheduled syncs via APScheduler
    - Optional real-time file monitoring via watchdog
    - Debounced sync triggers (30 second window)
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._observer = None
        self._running = False
        self._debounce_task = None
        self._debounce_lock = asyncio.Lock()

    async def start(
        self,
        watch_dirs: list[str] | None = None,
        cron_expression: str = "0 2 * * *",
    ) -> None:
        """Start the scheduler with cron job and optional file watching.

        Args:
            watch_dirs: Directories to monitor for file changes (optional).
            cron_expression: Cron expression for scheduled syncs.
                Default: "0 2 * * *" (2 AM daily)
                Format: minute hour day month day_of_week
        """
        if self._running:
            logger.warning("Scheduler already running")
            return

        # Parse cron expression
        parts = cron_expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expression}")

        minute, hour, day, month, day_of_week = parts

        # Create cron trigger
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
        )

        # Add scheduled job
        self.scheduler.add_job(
            self._run_sync,
            trigger=trigger,
            id="scheduled_sync",
            name="Scheduled Document Sync",
            replace_existing=True,
        )

        # Start scheduler
        self.scheduler.start()
        self._running = True

        logger.info(
            "Scheduler started with cron: %s (next run: %s)",
            cron_expression,
            self.get_status()["next_run_time"],
        )

        # Start file watching if requested and available
        if watch_dirs and WATCHDOG_AVAILABLE:
            await self._start_watching(watch_dirs)
        elif watch_dirs and not WATCHDOG_AVAILABLE:
            logger.warning("File watching requested but watchdog not available")

    async def stop(self) -> None:
        """Stop the scheduler and file watching."""
        if not self._running:
            return

        # Stop watchdog observer
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            logger.info("File watching stopped")

        # Shutdown scheduler
        self.scheduler.shutdown(wait=False)
        self._running = False

        # Cancel any pending debounce
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        logger.info("Scheduler stopped")

    async def _run_sync(self) -> None:
        """Execute a document sync operation."""
        logger.info("Starting scheduled document sync")

        try:
            from pipeline.sync import get_sync_engine
            engine = get_sync_engine()
            result = await engine.run_sync()

            logger.info(
                "Sync completed - added: %d, updated: %d, deleted: %d, failed: %d",
                result.get("added", 0),
                result.get("updated", 0),
                result.get("deleted", 0),
                result.get("failed", 0),
            )

        except ImportError:
            logger.error("SyncEngine not available - sync skipped")
        except Exception as e:
            logger.error("Sync failed: %s", e, exc_info=True)

    async def trigger_now(self) -> None:
        """Manually trigger a sync immediately."""
        logger.info("Manual sync trigger")
        await self._run_sync()

    def get_status(self) -> dict:
        """Get scheduler status.

        Returns:
            Dict with running, next_run_time, job_count
        """
        next_run = None
        job = self.scheduler.get_job("scheduled_sync")

        if job and job.next_run_time:
            next_run = job.next_run_time.isoformat()

        return {
            "running": self._running,
            "next_run_time": next_run,
            "job_count": len(self.scheduler.get_jobs()),
        }

    async def _start_watching(self, watch_dirs: list[str]) -> None:
        """Start watchdog file system monitoring."""
        if not WATCHDOG_AVAILABLE:
            return

        self._observer = Observer()
        handler = self._FileChangeHandler(self)

        for dir_path in watch_dirs:
            path = Path(dir_path)
            if not path.exists():
                logger.warning("Watch directory does not exist: %s", dir_path)
                continue

            self._observer.schedule(handler, str(path), recursive=True)
            logger.info("Watching directory: %s", dir_path)

        self._observer.start()

    async def _debounced_sync(self) -> None:
        """Trigger sync after debounce period (30 seconds)."""
        async with self._debounce_lock:
            # Cancel existing debounce task if any
            if self._debounce_task and not self._debounce_task.done():
                self._debounce_task.cancel()

            # Schedule new sync after 30 seconds
            async def delayed_sync():
                await asyncio.sleep(30)
                await self._run_sync()

            self._debounce_task = asyncio.create_task(delayed_sync())

    class _FileChangeHandler:
        """Watchdog event handler for file system changes."""

        def __init__(self, scheduler: "SyncScheduler"):
            if WATCHDOG_AVAILABLE:
                # Only inherit if watchdog is available
                FileSystemEventHandler.__init__(self)
            self.scheduler = scheduler

        def on_created(self, event: "FileSystemEvent") -> None:
            """Handle file creation."""
            if event.is_directory:
                return
            self._handle_change(event.src_path, "created")

        def on_modified(self, event: "FileSystemEvent") -> None:
            """Handle file modification."""
            if event.is_directory:
                return
            self._handle_change(event.src_path, "modified")

        def on_deleted(self, event: "FileSystemEvent") -> None:
            """Handle file deletion."""
            if event.is_directory:
                return
            self._handle_change(event.src_path, "deleted")

        def _handle_change(self, path: str, event_type: str) -> None:
            """Process file change event with filtering and debouncing."""
            file_path = Path(path)

            # Filter by allowed extensions
            if file_path.suffix.lower() not in settings.allowed_extensions:
                return

            logger.info("File %s detected: %s", event_type, file_path.name)

            # Trigger debounced sync
            asyncio.create_task(self.scheduler._debounced_sync())


# Singleton instance
_scheduler: SyncScheduler | None = None


def get_scheduler() -> SyncScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SyncScheduler()
    return _scheduler
