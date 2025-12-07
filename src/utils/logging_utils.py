# src/utils/logging_utils.py
"""
Centralized logging configuration for CARA-WR.

Provides consistent, informative logging across all modules with support for:
- Console output with colors (if available)
- File logging with timestamps
- Progress tracking for long-running operations
- Configurable verbosity levels
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import time


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = 'cara_wr',
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    use_colors: bool = True,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the CARA-WR project.
    
    Args:
        name: Logger name (default: 'cara_wr')
        level: Logging level (default: INFO)
        log_dir: Directory for log files (optional)
        log_file: Specific log filename (optional, auto-generated if log_dir provided)
        use_colors: Whether to use colors in console output
        format_str: Custom format string (optional)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_str is None:
        format_str = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors and sys.stdout.isatty():
        console_formatter = ColorFormatter(format_str, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(format_str, datefmt=date_format)
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'{name}_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_str, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'cara_wr') -> logging.Logger:
    """Get or create a logger with the given name."""
    return logging.getLogger(name)


class ProgressLogger:
    """
    A simple progress logger for long-running operations.
    
    Usage:
        progress = ProgressLogger(total=100, desc='Processing')
        for i in range(100):
            # do work
            progress.update()
        progress.finish()
    """
    
    def __init__(
        self,
        total: int,
        desc: str = 'Progress',
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10  # Log every N% progress
    ):
        self.total = total
        self.desc = desc
        self.logger = logger or get_logger()
        self.log_interval = log_interval
        
        self.current = 0
        self.start_time = time.time()
        self.last_log_pct = 0
        
        self.logger.info(f'🚀 Starting: {desc} ({total} items)')
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        pct = int(100 * self.current / self.total)
        
        # Log at intervals
        if pct >= self.last_log_pct + self.log_interval:
            elapsed = time.time() - self.start_time
            items_per_sec = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
            
            self.logger.info(
                f'   [{pct:3d}%] {self.current:,}/{self.total:,} | '
                f'{items_per_sec:.1f} it/s | '
                f'ETA: {remaining:.0f}s'
            )
            self.last_log_pct = pct
    
    def finish(self):
        """Mark operation as complete and log summary."""
        elapsed = time.time() - self.start_time
        items_per_sec = self.total / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f'✅ Completed: {self.desc} | '
            f'{self.total:,} items in {elapsed:.1f}s ({items_per_sec:.1f} it/s)'
        )


class Timer:
    """
    Context manager for timing operations.
    
    Usage:
        with Timer('Training epoch', logger):
            # do work
    """
    
    def __init__(self, desc: str, logger: Optional[logging.Logger] = None):
        self.desc = desc
        self.logger = logger or get_logger()
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f'⏱️  Starting: {self.desc}')
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        self.logger.info(f'⏱️  Finished: {self.desc} ({self.elapsed:.2f}s)')


def log_config(config: dict, logger: Optional[logging.Logger] = None, title: str = 'Configuration'):
    """Log a configuration dictionary in a formatted way."""
    logger = logger or get_logger()
    
    logger.info(f'📋 {title}:')
    max_key_len = max(len(str(k)) for k in config.keys())
    
    for key, value in config.items():
        logger.info(f'   {str(key):<{max_key_len}}: {value}')


def log_metrics(metrics: dict, logger: Optional[logging.Logger] = None, title: str = 'Metrics'):
    """Log metrics in a formatted box."""
    logger = logger or get_logger()
    
    logger.info('')
    logger.info(f'📊 {title}:')
    logger.info('   ╔════════════════════════════════════╗')
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if value < 1:  # Assume it's a ratio, convert to percentage
                logger.info(f'   ║  {key:<12}: {value * 100:6.2f}%           ║')
            else:
                logger.info(f'   ║  {key:<12}: {value:10.4f}         ║')
        else:
            logger.info(f'   ║  {key:<12}: {str(value):>10}         ║')
    
    logger.info('   ╚════════════════════════════════════╝')


def log_separator(char: str = '=', length: int = 70, logger: Optional[logging.Logger] = None):
    """Log a separator line."""
    logger = logger or get_logger()
    logger.info(char * length)


def log_header(title: str, logger: Optional[logging.Logger] = None, char: str = '='):
    """Log a header with separators."""
    logger = logger or get_logger()
    length = 70
    
    logger.info('')
    logger.info(char * length)
    logger.info(f'  {title}')
    logger.info(char * length)
