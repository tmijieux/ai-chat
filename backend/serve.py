import asyncio
import sys
import logging
import subprocess
import os

logger = logging.getLogger(__name__)

_RELOADER_ENV_VAR = "HYPERCORN_RELOADER"


class _ColorFormatter(logging.Formatter):
    """Colors log lines by level, with pipeline/workflow loggers given their own color at INFO/DEBUG
    so stage lifecycle logs stand out from the rest of the wall of text. WARNING/ERROR always win
    regardless of logger, since those need to stay visible no matter where they came from.
    """

    _LEVEL_COLORS = {
        logging.WARNING: "\x1b[33m",    # yellow
        logging.ERROR: "\x1b[31m",      # red
        logging.CRITICAL: "\x1b[1;31m", # bold red
    }
    _DEFAULT_COLORS = {
        logging.DEBUG: "\x1b[2m",  # dim
        logging.INFO: "\x1b[36m",  # cyan
    }
    _LOGGER_COLORS = {
        "agent.pipeline": "\x1b[95m",        # bright magenta
        "agent.custom_workflow": "\x1b[94m", # bright blue
    }
    _RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno in self._LEVEL_COLORS:
            color = self._LEVEL_COLORS[record.levelno]
        else:
            color = self._LOGGER_COLORS.get(record.name, self._DEFAULT_COLORS.get(record.levelno, ""))
        return f"{color}{message}{self._RESET}" if color else message


def _setup_logging(level: int) -> None:
    """Set up colored console logging. Must be called before any other logging.basicConfig()."""
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter("%(levelname)s:%(name)s:%(message)s"))
    logging.basicConfig(level=level, handlers=[handler])


def _is_running_in_reloader() -> bool:
    """Check if this process was spawned by the reloader parent."""
    return os.environ.get(_RELOADER_ENV_VAR, "") == "yes"


def _get_reloader_args() -> list[str]:
    """Return the argv needed to re-execute this script in a new process."""
    return [sys.executable, *sys.orig_argv[1:]]


async def _serve_with_watcher(host: str, port: int) -> bool:
    """Run Hypercorn alongside a watchfiles watcher.

    Returns True if shutdown was triggered by a file change (caller should exit 3),
    False if it was a clean shutdown (e.g. SIGINT).
    """
    from hypercorn.config import Config as HyperConfig
    from hypercorn.asyncio import serve
    from main import app

    config = HyperConfig()
    config.bind = [f"{host}:{port}"]
    config.accesslog = "-"
    config.errorlog = "-"

    shutdown_event = asyncio.Event()
    file_changed = False
    watch_dir = os.path.dirname(os.path.abspath(__file__))

    async def watch_and_signal() -> None:
        nonlocal file_changed
        from watchfiles import awatch, PythonFilter
        async for _ in awatch(watch_dir, watch_filter=PythonFilter()):
            file_changed = True
            shutdown_event.set()
            return

    watch_task = asyncio.create_task(watch_and_signal())
    try:
        await serve(app, config, shutdown_trigger=shutdown_event.wait)
    finally:
        watch_task.cancel()
        try:
            await watch_task
        except asyncio.CancelledError:
            pass

    return file_changed


def main() -> None:
    """Entry point — acts as either the reloader parent or the worker child."""
    _setup_logging(logging.INFO)

    if _is_running_in_reloader():
        try:
            reload_needed = asyncio.run(_serve_with_watcher("0.0.0.0", 8000))
        except KeyboardInterrupt:
            return
        if reload_needed:
            sys.exit(3)
    else:
        while True:
            logger.info(" * Restarting with hypercorn reloader")
            new_environ = os.environ.copy()
            new_environ[_RELOADER_ENV_VAR] = "yes"
            exit_code = subprocess.call(_get_reloader_args(), env=new_environ, close_fds=False)
            logger.info("Worker exited with code %s", exit_code)
            if exit_code != 3:
                return


if __name__ == "__main__":
    main()
