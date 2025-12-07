# src/utils/logging_config.py
"""
Enhanced logging configuration with colors, file output, and debug levels.

Features:
- Colored console output (works in terminals and notebooks)
- File logging for each phase/experiment
- Configurable debug levels
- Performance timing decorators
- Memory usage tracking
"""
import logging
import sys
import os
import time
import functools
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any
import traceback

# Try to import colorama for Windows compatibility
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# COLOR DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal/notebook output."""
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Bold variants
    BOLD = '\033[1m'
    BOLD_RED = '\033[1;91m'
    BOLD_GREEN = '\033[1;92m'
    BOLD_YELLOW = '\033[1;93m'
    BOLD_BLUE = '\033[1;94m'
    BOLD_CYAN = '\033[1;96m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    # Reset
    RESET = '\033[0m'
    ENDC = '\033[0m'
    
    # Semantic colors for logging levels
    DEBUG = GRAY
    INFO = CYAN
    SUCCESS = GREEN
    WARNING = YELLOW
    ERROR = RED
    CRITICAL = BOLD_RED
    
    # Semantic colors for phases
    PHASE_START = BOLD_BLUE
    PHASE_END = BOLD_GREEN
    METRIC = MAGENTA
    DATA = BLUE
    MODEL = CYAN
    TIMING = YELLOW


def colorize(text: str, color: str) -> str:
    """Apply color to text."""
    return f"{color}{text}{Colors.RESET}"


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM FORMATTER WITH COLORS
# ═══════════════════════════════════════════════════════════════════════════

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.BOLD_RED,
    }
    
    LEVEL_ICONS = {
        logging.DEBUG: '🔍',
        logging.INFO: 'ℹ️ ',
        logging.WARNING: '⚠️ ',
        logging.ERROR: '❌',
        logging.CRITICAL: '🚨',
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        # Save original values
        orig_msg = record.msg
        orig_levelname = record.levelname
        
        if self.use_colors:
            color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
            icon = self.LEVEL_ICONS.get(record.levelno, '')
            
            # Colorize level name
            record.levelname = f"{color}{record.levelname:8s}{Colors.RESET}"
            
            # Add icon to message
            if not str(record.msg).startswith(('🔍', 'ℹ️', '⚠️', '❌', '🚨', '📊', '🎯', '💾', '🔄', '✅', '📁', '🏗️', '📈', '📉')):
                record.msg = f"{icon} {record.msg}"
        
        result = super().format(record)
        
        # Restore original values
        record.msg = orig_msg
        record.levelname = orig_levelname
        
        return result


class PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no colors)."""
    
    LEVEL_ICONS = {
        logging.DEBUG: '[DEBUG]',
        logging.INFO: '[INFO]',
        logging.WARNING: '[WARN]',
        logging.ERROR: '[ERROR]',
        logging.CRITICAL: '[CRIT]',
    }
    
    def format(self, record):
        # Add level icon
        icon = self.LEVEL_ICONS.get(record.levelno, '[LOG]')
        record.levelname = icon
        return super().format(record)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE LOGGER - For structured experiment logging
# ═══════════════════════════════════════════════════════════════════════════

class PhaseLogger:
    """
    Structured logger for experiment phases with timing and file output.
    
    Usage:
        logger = PhaseLogger('experiment_name', log_dir='logs/')
        with logger.phase('Training'):
            # training code
            logger.metric('loss', 0.5)
            logger.debug('Batch processed', batch_size=32)
    
    Note: This also configures the root logger so that all module-level loggers
    (e.g., logging.getLogger(__name__)) inherit the same handlers.
    """
    
    def __init__(self, name: str, log_dir: Optional[str] = None, 
                 level: int = logging.DEBUG, console_level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        self.logger.propagate = False

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_fmt = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_fmt)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_dir specified)
        self.log_file = None
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = log_path / f'{name}_{timestamp}.log'
            
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(level)
            file_fmt = PlainFormatter(
                fmt='%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_fmt)
            self.logger.addHandler(file_handler)
        
        # ════════════════════════════════════════════════════════════════
        # IMPORTANT: Also configure the root logger so that module-level
        # loggers (e.g., logging.getLogger(__name__) in scripts) inherit
        # the same handlers. This prevents duplicate logging and ensures
        # all logs go to the same file.
        # ════════════════════════════════════════════════════════════════
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.handlers = []  # Clear any existing handlers
        
        # Add same handlers to root logger
        root_console_handler = logging.StreamHandler(sys.stdout)
        root_console_handler.setLevel(console_level)
        root_console_handler.setFormatter(console_fmt)
        root_logger.addHandler(root_console_handler)
        
        if log_dir and self.log_file:
            root_file_handler = logging.FileHandler(self.log_file)
            root_file_handler.setLevel(level)
            root_file_handler.setFormatter(file_fmt)
            root_logger.addHandler(root_file_handler)
        
        # Phase tracking
        self._current_phase = None
        self._phase_start_time = None
        self._phase_stack = []
        
        # Metrics tracking
        self._metrics = {}
    
    def phase(self, name: str):
        """Context manager for a named phase."""
        return _PhaseContext(self, name)
    
    def start_phase(self, name: str):
        """Start a named phase."""
        self._phase_stack.append((self._current_phase, self._phase_start_time))
        self._current_phase = name
        self._phase_start_time = time.time()
        
        header = f"{'═' * 60}"
        self.logger.info(f"\n{Colors.PHASE_START}{header}{Colors.RESET}")
        self.logger.info(f"{Colors.PHASE_START}🚀 STARTING PHASE: {name}{Colors.RESET}")
        self.logger.info(f"{Colors.PHASE_START}{header}{Colors.RESET}")
    
    def end_phase(self, name: str = None):
        """End the current phase."""
        if name and name != self._current_phase:
            self.logger.warning(f"Phase mismatch: ending '{name}' but current is '{self._current_phase}'")
        
        elapsed = time.time() - self._phase_start_time
        
        self.logger.info(f"{Colors.PHASE_END}✅ COMPLETED: {self._current_phase} ({elapsed:.2f}s){Colors.RESET}")
        self.logger.info(f"{Colors.PHASE_END}{'─' * 60}{Colors.RESET}\n")
        
        # Restore previous phase
        if self._phase_stack:
            self._current_phase, self._phase_start_time = self._phase_stack.pop()
        else:
            self._current_phase = None
            self._phase_start_time = None
        
        return elapsed
    
    # Logging methods
    def debug(self, msg: str, **kwargs):
        """Debug level log with optional key-value pairs."""
        if kwargs:
            details = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
            msg = f"{msg} | {Colors.GRAY}{details}{Colors.RESET}"
        self.logger.debug(msg)
    
    def info(self, msg: str, **kwargs):
        """Info level log."""
        if kwargs:
            details = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
            msg = f"{msg} | {details}"
        self.logger.info(msg)
    
    def success(self, msg: str):
        """Success message (info level with green color)."""
        self.logger.info(f"{Colors.SUCCESS}✅ {msg}{Colors.RESET}")
    
    def warning(self, msg: str, **kwargs):
        """Warning level log."""
        if kwargs:
            details = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
            msg = f"{msg} | {details}"
        self.logger.warning(msg)
    
    def error(self, msg: str, exc_info: bool = False, **kwargs):
        """Error level log."""
        if kwargs:
            details = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
            msg = f"{msg} | {details}"
        self.logger.error(msg, exc_info=exc_info)
    
    def critical(self, msg: str, exc_info: bool = True):
        """Critical level log."""
        self.logger.critical(msg, exc_info=exc_info)
    
    def metric(self, name: str, value: Any, step: int = None):
        """Log a metric value."""
        self._metrics.setdefault(name, []).append((step, value))
        step_str = f"[step {step}] " if step is not None else ""
        self.logger.info(f"{Colors.METRIC}📊 METRIC: {step_str}{name} = {value}{Colors.RESET}")
    
    def data(self, msg: str, **kwargs):
        """Log data-related information."""
        if kwargs:
            details = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
            msg = f"{msg} | {details}"
        self.logger.info(f"{Colors.DATA}📁 DATA: {msg}{Colors.RESET}")
    
    def model(self, msg: str, **kwargs):
        """Log model-related information."""
        if kwargs:
            details = ' | '.join(f'{k}={v}' for k, v in kwargs.items())
            msg = f"{msg} | {details}"
        self.logger.info(f"{Colors.MODEL}🏗️  MODEL: {msg}{Colors.RESET}")
    
    def timing(self, msg: str, elapsed: float):
        """Log timing information."""
        self.logger.info(f"{Colors.TIMING}⏱️  TIMING: {msg} ({elapsed:.3f}s){Colors.RESET}")
    
    def tensor_info(self, name: str, tensor, show_stats: bool = True):
        """Log tensor shape and statistics for debugging."""
        import torch
        import numpy as np
        
        if isinstance(tensor, torch.Tensor):
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            device = tensor.device
            if show_stats and tensor.numel() > 0:
                stats = f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}"
            else:
                stats = "N/A"
        elif isinstance(tensor, np.ndarray):
            shape = tensor.shape
            dtype = tensor.dtype
            device = "cpu"
            if show_stats and tensor.size > 0:
                stats = f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}"
            else:
                stats = "N/A"
        else:
            shape = "unknown"
            dtype = type(tensor)
            device = "N/A"
            stats = "N/A"
        
        self.debug(f"TENSOR '{name}': shape={shape}, dtype={dtype}, device={device}, stats=[{stats}]")
    
    def separator(self, char: str = '─', length: int = 70):
        """Print a separator line."""
        self.logger.info(char * length)
    
    def header(self, title: str, char: str = '═', length: int = 70):
        """Print a header with title."""
        self.logger.info(f"\n{char * length}")
        self.logger.info(f"{Colors.BOLD}{title.center(length)}{Colors.RESET}")
        self.logger.info(f"{char * length}")
    
    def table(self, headers: list, rows: list, title: str = None):
        """Print a formatted table."""
        if title:
            self.info(f"\n{title}")
        
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))
        
        # Format header
        header_str = " │ ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        separator = "─┼─".join("─" * w for w in widths)
        
        self.info(f"┌─{'─┬─'.join('─' * w for w in widths)}─┐")
        self.info(f"│ {header_str} │")
        self.info(f"├─{separator}─┤")
        
        for row in rows:
            row_str = " │ ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            self.info(f"│ {row_str} │")
        
        self.info(f"└─{'─┴─'.join('─' * w for w in widths)}─┘")


class _PhaseContext:
    """Context manager for phases."""
    
    def __init__(self, logger: PhaseLogger, name: str):
        self.logger = logger
        self.name = name
    
    def __enter__(self):
        self.logger.start_phase(self.name)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Phase '{self.name}' failed: {exc_val}", exc_info=True)
        self.logger.end_phase(self.name)
        return False


# ═══════════════════════════════════════════════════════════════════════════
# DECORATORS FOR TIMING AND DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════

def timed(func: Callable = None, *, logger: logging.Logger = None, level: int = logging.DEBUG):
    """
    Decorator to time function execution.
    
    Usage:
        @timed
        def my_function():
            ...
        
        @timed(logger=my_logger, level=logging.INFO)
        def my_function():
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.time() - start
                msg = f"⏱️  {fn.__module__}.{fn.__name__}() completed in {elapsed:.3f}s"
                if logger:
                    logger.log(level, msg)
                else:
                    logging.log(level, msg)
                return result
            except Exception as e:
                elapsed = time.time() - start
                msg = f"❌ {fn.__module__}.{fn.__name__}() failed after {elapsed:.3f}s: {e}"
                if logger:
                    logger.error(msg)
                else:
                    logging.error(msg)
                raise
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def debug_call(func: Callable = None, *, logger: logging.Logger = None, 
               log_args: bool = True, log_result: bool = False):
    """
    Decorator to log function calls with arguments.
    
    Usage:
        @debug_call
        def my_function(x, y):
            return x + y
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            func_name = f"{fn.__module__}.{fn.__name__}"
            
            if log_args:
                args_repr = ', '.join(repr(a)[:50] for a in args)
                kwargs_repr = ', '.join(f'{k}={repr(v)[:30]}' for k, v in kwargs.items())
                all_args = ', '.join(filter(None, [args_repr, kwargs_repr]))
                msg = f"🔍 CALL: {func_name}({all_args[:200]})"
            else:
                msg = f"🔍 CALL: {func_name}()"
            
            if logger:
                logger.debug(msg)
            else:
                logging.debug(msg)
            
            result = fn(*args, **kwargs)
            
            if log_result:
                result_repr = repr(result)[:100]
                msg = f"🔍 RETURN: {func_name} -> {result_repr}"
                if logger:
                    logger.debug(msg)
                else:
                    logging.debug(msg)
            
            return result
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# MEMORY TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def log_memory_usage(logger: logging.Logger = None, prefix: str = ""):
    """Log current memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            msg = f"🧠 GPU Memory: {prefix}allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_allocated:.2f}GB"
        else:
            msg = f"🧠 Memory: CUDA not available"
    except ImportError:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        msg = f"🧠 Memory: {prefix}RSS={mem.rss/1e9:.2f}GB, VMS={mem.vms/1e9:.2f}GB"
    except:
        msg = f"🧠 Memory: Unable to get memory info"
    
    if logger:
        logger.debug(msg)
    else:
        logging.debug(msg)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL LOGGER SETUP
# ═══════════════════════════════════════════════════════════════════════════

_global_logger: Optional[PhaseLogger] = None


def setup_logging(name: str = 'cara_wr', 
                  log_dir: Optional[str] = None,
                  level: int = logging.DEBUG,
                  console_level: int = logging.INFO) -> PhaseLogger:
    """
    Setup global logging configuration.
    
    Args:
        name: Logger name
        log_dir: Directory for log files (None for console only)
        level: File logging level
        console_level: Console logging level
        
    Returns:
        PhaseLogger instance
    """
    global _global_logger
    _global_logger = PhaseLogger(name, log_dir, level, console_level)
    return _global_logger


def get_logger() -> PhaseLogger:
    """Get the global logger (creates one if not exists)."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK-SPECIFIC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def setup_notebook_logging(experiment_name: str, log_dir: str = 'logs',
                          debug: bool = False) -> PhaseLogger:
    """
    Setup logging specifically for Jupyter notebooks.
    
    Args:
        experiment_name: Name for this experiment run
        log_dir: Directory for log files
        debug: If True, show debug messages in console
        
    Returns:
        PhaseLogger configured for notebook use
    """
    console_level = logging.DEBUG if debug else logging.INFO
    logger = setup_logging(
        name=experiment_name,
        log_dir=log_dir,
        level=logging.DEBUG,
        console_level=console_level
    )
    
    # Print startup message
    logger.header(f"🚀 {experiment_name.upper()}")
    logger.info(f"📁 Log file: {logger.log_file}")
    logger.info(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.separator()
    
    return logger


def setup_script_logging(script_name: str, log_dir: Optional[str] = None,
                         debug: bool = False) -> PhaseLogger:
    """
    Setup logging for command-line scripts.
    
    Args:
        script_name: Name of the script (e.g., 'train_resnet_triplet')
        log_dir: Directory for log files. If None, logs go to 'logs/' in current dir
        debug: If True, show debug messages in console too
        
    Returns:
        PhaseLogger configured for script use
        
    Example:
        logger = setup_script_logging('train', log_dir='experiments/cvl/logs')
        logger.info('Starting training...')
        logger.debug('Detailed info here')  # Only in file unless debug=True
    """
    if log_dir is None:
        log_dir = 'logs'
    
    console_level = logging.DEBUG if debug else logging.INFO
    logger = setup_logging(
        name=script_name,
        log_dir=log_dir,
        level=logging.DEBUG,  # Always save DEBUG to file
        console_level=console_level
    )
    
    logger.info(f"📁 Log file: {logger.log_file}")
    logger.info(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def get_log_dir_for_experiment(experiment_type: str, base_dir: str = 'experiments') -> Path:
    """
    Get the log directory for a specific experiment type.
    
    Args:
        experiment_type: Type of experiment (e.g., 'cvl', 'iam', 'hisir19')
        base_dir: Base experiments directory
        
    Returns:
        Path to log directory (creates if not exists)
        
    Example:
        log_dir = get_log_dir_for_experiment('cvl_pages')
        # Returns: experiments/cvl_pages/logs/
    """
    log_path = Path(base_dir) / experiment_type / 'logs'
    log_path.mkdir(parents=True, exist_ok=True)
    return log_path


def print_config(config: dict, title: str = "Configuration", logger: PhaseLogger = None):
    """Pretty print a configuration dictionary."""
    log = logger or get_logger()
    log.header(title)
    max_key_len = max(len(str(k)) for k in config.keys())
    for key, value in config.items():
        log.info(f"  {str(key).ljust(max_key_len)} : {value}")
    log.separator()
