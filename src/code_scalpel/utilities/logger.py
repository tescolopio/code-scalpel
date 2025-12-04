import logging
import logging.handlers
import threading
import time
from contextlib import contextmanager  # Import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from queue import Queue
from typing import Any, Optional


class LogLevel(Enum):
    """Custom logging levels."""

    TRACE = 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """Configuration for logger."""

    name: str
    level: LogLevel = LogLevel.INFO
    file_path: Optional[str] = None
    console_output: bool = True
    rotation_size: Optional[int] = None  # bytes
    rotation_time: Optional[str] = None  # e.g., "midnight", "h", "d"
    max_files: Optional[int] = None
    format_string: Optional[str] = None
    include_timestamp: bool = True
    include_thread: bool = True
    include_process: bool = True
    include_function: bool = True
    include_line: bool = True
    enable_colors: bool = True
    enable_async: bool = False
    enable_remote: bool = False
    remote_host: Optional[str] = None
    remote_port: Optional[int] = None


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors to log messages."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        if not record.exc_info:
            level_color = self.COLORS.get(record.levelno)
            record.msg = f"{level_color}{record.msg}{self.RESET}"
        return super().format(record)


class AsyncHandler(logging.Handler):
    """Asynchronous log handler."""

    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.queue = Queue()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def emit(self, record):
        self.queue.put(record)

    def _process_queue(self):
        while True:
            try:
                record = self.queue.get()
                self.handler.emit(record)
            except Exception:
                pass
            finally:
                self.queue.task_done()


class ContextFilter(logging.Filter):
    """Filter for adding contextual information to log records."""

    def __init__(self, context: dict[str, Any]):
        super().__init__()
        self.context = context

    def filter(self, record):
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class EnhancedLogger:
    """Advanced logger with multiple outputs and features."""

    def __init__(self, config: LogConfig):
        self.config = config
        self.logger = logging.getLogger(config.name)
        self.context: dict[str, Any] = {}
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with all configured handlers."""
        # Set base level
        self.logger.setLevel(self.config.level.value)

        # Create formatter
        formatter = self._create_formatter()

        # Setup handlers
        handlers = []

        # File handler
        if self.config.file_path:
            file_handler = self._setup_file_handler(formatter)
            handlers.append(file_handler)

        # Console handler
        if self.config.console_output:
            console_handler = self._setup_console_handler(formatter)
            handlers.append(console_handler)

        # Remote handler
        if self.config.enable_remote:
            remote_handler = self._setup_remote_handler(formatter)
            handlers.append(remote_handler)

        # Apply async wrapper if enabled
        if self.config.enable_async:
            handlers = [AsyncHandler(h) for h in handlers]

        # Add handlers to logger
        for handler in handlers:
            self.logger.addHandler(handler)

        # Add context filter
        self.logger.addFilter(ContextFilter(self.context))

    def _create_formatter(self) -> logging.Formatter:
        """Create log formatter based on configuration."""
        parts = []

        if self.config.include_timestamp:
            parts.append("%(asctime)s")
        if self.config.include_process:
            parts.append("%(process)d")
        if self.config.include_thread:
            parts.append("%(threadName)s")
        if self.config.include_function:
            parts.append("%(funcName)s")
        if self.config.include_line:
            parts.append("%(lineno)d")

        parts.extend(["%(levelname)s", "%(name)s", "%(message)s"])

        format_string = self.config.format_string or " - ".join(parts)

        if self.config.enable_colors:
            return ColorFormatter(format_string)
        return logging.Formatter(format_string)

    def _setup_file_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """Setup file handler with rotation if configured."""
        if self.config.rotation_size:
            handler = logging.handlers.RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.rotation_size,
                backupCount=self.config.max_files or 5,
            )
        elif self.config.rotation_time:
            handler = logging.handlers.TimedRotatingFileHandler(
                self.config.file_path,
                when=self.config.rotation_time,
                backupCount=self.config.max_files or 5,
            )
        else:
            handler = logging.FileHandler(self.config.file_path)

        handler.setFormatter(formatter)
        return handler

    def _setup_console_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """Setup console handler."""
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        return handler

    def _setup_remote_handler(self, formatter: logging.Formatter) -> logging.Handler:
        """Setup remote logging handler."""
        handler = logging.handlers.SysLogHandler(
            address=(
                self.config.remote_host or "localhost",
                self.config.remote_port or 514,
            )
        )
        handler.setFormatter(formatter)
        return handler

    def set_level(self, level: LogLevel):
        """Change logging level."""
        self.logger.setLevel(level.value)

    def add_context(self, **kwargs):
        """Add context to all log messages."""
        self.context.update(kwargs)

    def remove_context(self, *keys):
        """Remove context keys."""
        for key in keys:
            self.context.pop(key, None)

    @contextmanager
    def log_context(self, **kwargs):
        """Context manager for temporary context."""
        old_context = self.context.copy()
        self.add_context(**kwargs)
        try:
            yield
        finally:
            self.context = old_context

    def trace(self, msg, *args, **kwargs):
        """Log at TRACE level."""
        self.logger.log(LogLevel.TRACE.value, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Log at DEBUG level."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log at INFO level."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log at WARNING level."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log at ERROR level."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        """Log at CRITICAL level."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        """Log an exception with traceback."""
        self.logger.exception(msg, *args, exc_info=exc_info, **kwargs)

    def log_function(self, level: LogLevel = LogLevel.INFO):
        """Decorator to log function calls."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                self.log(
                    level.value,
                    f"Entering {func.__name__}",
                    extra={"args": args, "kwargs": kwargs},
                )
                try:
                    result = func(*args, **kwargs)
                    self.log(
                        level.value,
                        f"Exiting {func.__name__}",
                        extra={"duration": time.time() - start_time, "result": result},
                    )
                    return result
                except Exception as e:
                    self.exception(
                        f"Error in {func.__name__}: {str(e)}",
                        extra={"duration": time.time() - start_time},
                    )
                    raise

            return wrapper

        return decorator


def create_logger(
    name: str, level: LogLevel = LogLevel.INFO, **kwargs
) -> EnhancedLogger:
    """Create a new logger instance."""
    config = LogConfig(name=name, level=level, **kwargs)
    return EnhancedLogger(config)
