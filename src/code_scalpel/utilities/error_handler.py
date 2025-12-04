from typing import Dict, List, Set, Optional, Union, Any, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
import sys
from datetime import datetime
import json
from pathlib import Path
import contextlib
import threading
from functools import wraps
import warnings

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    FATAL = 'fatal'

class ErrorCategory(Enum):
    """Categories of errors."""
    PARSING = 'parsing'
    ANALYSIS = 'analysis'
    VALIDATION = 'validation'
    EXECUTION = 'execution'
    CONFIGURATION = 'configuration'
    RESOURCE = 'resource'
    SYSTEM = 'system'

@dataclass
class ErrorContext:
    """Context information for errors."""
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column: Optional[int] = None
    function_name: Optional[str] = None
    code_snippet: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorReport:
    """Comprehensive error report."""
    error_code: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    suggestions: List[str] = field(default_factory=list)
    related_errors: List['ErrorReport'] = field(default_factory=list)

class CodeScalpelError(Exception):
    """Base exception class for Code Scalpel."""
    
    def __init__(self, message: str,
                 error_code: str,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 context: Optional[ErrorContext] = None,
                 suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.suggestions = suggestions or []

class ParsingError(CodeScalpelError):
    """Errors during code parsing."""
    def __init__(self, message: str, error_code: str = 'PARSE001', **kwargs):
        super().__init__(
            message,
            error_code,
            category=ErrorCategory.PARSING,
            **kwargs
        )

class AnalysisError(CodeScalpelError):
    """Errors during code analysis."""
    def __init__(self, message: str, error_code: str = 'ANALYZE001', **kwargs):
        super().__init__(
            message,
            error_code,
            category=ErrorCategory.ANALYSIS,
            **kwargs
        )

class ValidationError(CodeScalpelError):
    """Errors during validation."""
    def __init__(self, message: str, error_code: str = 'VALID001', **kwargs):
        super().__init__(
            message,
            error_code,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )

class ConfigurationError(CodeScalpelError):
    """Configuration-related errors."""
    def __init__(self, message: str, error_code: str = 'CONFIG001', **kwargs):
        super().__init__(
            message,
            error_code,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )

class ResourceError(CodeScalpelError):
    """Resource-related errors."""
    def __init__(self, message: str, error_code: str = 'RESOURCE001', **kwargs):
        super().__init__(
            message,
            error_code,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )

class ErrorHandler:
    """Advanced error handler with reporting and recovery capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_history: List[ErrorReport] = []
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = defaultdict(list)
        self.recovery_strategies: Dict[str, Callable] = {}
        self._setup_logging()
        self._load_error_codes()

    def handle_error(self, error: Exception,
                    raise_error: bool = True,
                    collect_context: bool = True) -> Optional[ErrorReport]:
        """
        Handle an error with comprehensive error management.
        
        Args:
            error: The error to handle
            raise_error: Whether to raise the error after handling
            collect_context: Whether to collect context information
            
        Returns:
            Error report if error was handled
        """
        try:
            # Create error report
            report = self._create_error_report(error, collect_context)
            
            # Log error
            self._log_error(report)
            
            # Store in history
            self.error_history.append(report)
            
            # Notify callbacks
            self._notify_callbacks(report)
            
            # Attempt recovery
            if self._should_attempt_recovery(report):
                if self._attempt_recovery(report):
                    return report
            
            # Raise if configured
            if raise_error:
                raise error
                
            return report
            
        except Exception as e:
            # Handle errors in error handling
            logging.error(f"Error in error handler: {str(e)}")
            if raise_error:
                raise

    def register_callback(self, category: ErrorCategory,
                         callback: Callable[[ErrorReport], None]):
        """Register callback for error category."""
        self.error_callbacks[category].append(callback)

    def register_recovery_strategy(self, error_code: str,
                                 strategy: Callable[[ErrorReport], bool]):
        """Register recovery strategy for error code."""
        self.recovery_strategies[error_code] = strategy

    def get_error_history(self, 
                         category: Optional[ErrorCategory] = None,
                         severity: Optional[ErrorSeverity] = None) -> List[ErrorReport]:
        """Get filtered error history."""
        filtered = self.error_history
        
        if category:
            filtered = [e for e in filtered if e.category == category]
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
            
        return filtered

    def clear_history(self):
        """Clear error history."""
        self.error_history.clear()

    def export_error_report(self, file_path: str):
        """Export error history to file."""
        reports = [
            {
                'error_code': report.error_code,
                'message': report.message,
                'severity': report.severity.value,
                'category': report.category.value,
                'timestamp': report.context.timestamp.isoformat(),
                'context': {
                    'file_path': report.context.file_path,
                    'line_number': report.context.line_number,
                    'column': report.context.column,
                    'function_name': report.context.function_name,
                    'code_snippet': report.context.code_snippet,
                    'additional_info': report.context.additional_info
                },
                'suggestions': report.suggestions
            }
            for report in self.error_history
        ]
        
        with open(file_path, 'w') as f:
            json.dump(reports, f, indent=2)

    @contextlib.contextmanager
    def error_context(self, **context_info):
        """Context manager for error handling with additional context."""
        try:
            yield
        except Exception as e:
            if isinstance(e, CodeScalpelError):
                e.context = ErrorContext(**{
                    **e.context.__dict__,
                    **context_info
                })
            self.handle_error(e)

    def error_decorator(self, error_category: ErrorCategory):
        """Decorator for error handling."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if not isinstance(e, CodeScalpelError):
                        e = CodeScalpelError(
                            str(e),
                            'RUNTIME001',
                            category=error_category
                        )
                    self.handle_error(e)
            return wrapper
        return decorator

    def _create_error_report(self, error: Exception,
                           collect_context: bool) -> ErrorReport:
        """Create comprehensive error report."""
        if isinstance(error, CodeScalpelError):
            context = error.context
            if collect_context:
                self._enhance_context(context)
            
            return ErrorReport(
                error_code=error.error_code,
                message=str(error),
                severity=error.severity,
                category=error.category,
                context=context,
                suggestions=error.suggestions
            )
        else:
            # Create generic error report
            context = ErrorContext()
            if collect_context:
                self._enhance_context(context)
                
            return ErrorReport(
                error_code='RUNTIME001',
                message=str(error),
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.SYSTEM,
                context=context
            )

    def _enhance_context(self, context: ErrorContext):
        """Enhance error context with additional information."""
        # Get stack trace
        tb = traceback.extract_stack()
        context.stack_trace = ''.join(traceback.format_list(tb))
        
        # Get function name
        frame = sys._getframe(3)
        context.function_name = frame.f_code.co_name
        
        # Get code snippet if file path available
        if context.file_path and context.line_number:
            try:
                with open(context.file_path) as f:
                    lines = f.readlines()
                    start = max(0, context.line_number - 3)
                    end = min(len(lines), context.line_number + 3)
                    context.code_snippet = ''.join(lines[start:end])
            except Exception:
                pass

    def _log_error(self, report: ErrorReport):
        """Log error with appropriate severity."""
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }[report.severity]
        
        self.logger.log(
            log_level,
            f"[{report.error_code}] {report.message}",
            extra={
                'error_code': report.error_code,
                'category': report.category.value,
                'context': report.context.__dict__
            }
        )

    def _notify_callbacks(self, report: ErrorReport):
        """Notify registered callbacks."""
        for callback in self.error_callbacks[report.category]:
            try:
                callback(report)
            except Exception as e:
                logging.error(f"Error in callback: {str(e)}")

    def _should_attempt_recovery(self, report: ErrorReport) -> bool:
        """Determine if recovery should be attempted."""
        return (
            report.error_code in self.recovery_strategies and
            report.severity != ErrorSeverity.FATAL
        )

    def _attempt_recovery(self, report: ErrorReport) -> bool:
        """Attempt to recover from error."""
        strategy = self.recovery_strategies[report.error_code]
        try:
            return strategy(report)
        except Exception as e:
            logging.error(f"Error in recovery strategy: {str(e)}")
            return False

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(error_code)s] %(message)s'
        )
        self.logger = logging.getLogger('ErrorHandler')

    def _load_error_codes(self):
        """Load error codes and descriptions."""
        error_codes_file = self.config.get('error_codes_file', 'error_codes.json')
        try:
            with open(error_codes_file) as f:
                self.error_codes = json.load(f)
        except Exception:
            self.error_codes = {}

def create_error_handler(config: Optional[Dict[str, Any]] = None) -> ErrorHandler:
    """Create a new error handler instance."""
    return ErrorHandler(config)