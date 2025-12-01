"""
Sample Data Processing Library for benchmarking CodeAnalyzer.

This simulates a real-world data processing library with:
- Data structures
- Processing utilities  
- IO operations
- Some intentional dead code
"""

import json
import csv
import re
from typing import Dict, List, Optional, Any, Union, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import logging
from abc import ABC, abstractmethod
from functools import wraps, lru_cache
import threading

# Setup logging
logger = logging.getLogger(__name__)

# Type aliases
JsonDict = Dict[str, Any]
DataRow = Dict[str, Any]
FilterFunc = Callable[[DataRow], bool]


class DataFormat(Enum):
    """Supported data formats."""
    JSON = 'json'
    CSV = 'csv'
    TSV = 'tsv'
    XML = 'xml'


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ProcessingError(Exception):
    """Raised when data processing fails."""
    pass


@dataclass
class FieldSchema:
    """Schema for a data field."""
    name: str
    data_type: type
    required: bool = True
    default: Any = None
    validators: List[Callable] = field(default_factory=list)


@dataclass 
class DataSchema:
    """Schema for validating data."""
    fields: List[FieldSchema]
    strict: bool = True  # Reject unknown fields


@dataclass
class ProcessingStats:
    """Statistics from data processing."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def read(self) -> Iterator[DataRow]:
        pass
    
    @abstractmethod
    def write(self, data: Iterator[DataRow]) -> None:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass


class InMemoryDataSource(DataSource):
    """In-memory data source for testing."""
    
    def __init__(self, data: Optional[List[DataRow]] = None):
        self.data: List[DataRow] = data or []
        self._read_index = 0
    
    def read(self) -> Iterator[DataRow]:
        for row in self.data:
            yield row
    
    def write(self, data: Iterator[DataRow]) -> None:
        self.data.extend(data)
    
    def close(self) -> None:
        pass
    
    def __len__(self) -> int:
        return len(self.data)


class FileDataSource(DataSource):
    """File-based data source."""
    
    def __init__(self, filepath: str, format: DataFormat = DataFormat.JSON):
        self.filepath = Path(filepath)
        self.format = format
        self._file_handle = None
    
    def read(self) -> Iterator[DataRow]:
        if self.format == DataFormat.JSON:
            yield from self._read_json()
        elif self.format == DataFormat.CSV:
            yield from self._read_csv()
        else:
            raise ProcessingError(f"Unsupported format: {self.format}")
    
    def write(self, data: Iterator[DataRow]) -> None:
        if self.format == DataFormat.JSON:
            self._write_json(data)
        elif self.format == DataFormat.CSV:
            self._write_csv(data)
    
    def close(self) -> None:
        if self._file_handle:
            self._file_handle.close()
    
    def _read_json(self) -> Iterator[DataRow]:
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                yield from data
            else:
                yield data
    
    def _write_json(self, data: Iterator[DataRow]) -> None:
        with open(self.filepath, 'w') as f:
            json.dump(list(data), f, indent=2)
    
    def _read_csv(self) -> Iterator[DataRow]:
        with open(self.filepath, 'r', newline='') as f:
            reader = csv.DictReader(f)
            yield from reader
    
    def _write_csv(self, data: Iterator[DataRow]) -> None:
        data_list = list(data)
        if not data_list:
            return
        
        fieldnames = list(data_list[0].keys())
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
    
    # Dead code: unused method
    def deprecated_parse_xml(self) -> Iterator[DataRow]:
        """Old XML parsing method - deprecated."""
        raise NotImplementedError("XML parsing is deprecated")


class DataValidator:
    """Validate data against a schema."""
    
    def __init__(self, schema: DataSchema):
        self.schema = schema
        self._field_lookup = {f.name: f for f in schema.fields}
    
    def validate(self, row: DataRow) -> Tuple[bool, List[str]]:
        """Validate a single row."""
        errors = []
        
        # Check required fields
        for field in self.schema.fields:
            if field.required and field.name not in row:
                errors.append(f"Missing required field: {field.name}")
        
        # Check unknown fields in strict mode
        if self.schema.strict:
            known_fields = {f.name for f in self.schema.fields}
            for key in row:
                if key not in known_fields:
                    errors.append(f"Unknown field: {key}")
        
        # Validate field types and custom validators
        for field in self.schema.fields:
            if field.name in row:
                value = row[field.name]
                
                # Type check
                if value is not None and not isinstance(value, field.data_type):
                    try:
                        # Try to coerce
                        row[field.name] = field.data_type(value)
                    except (ValueError, TypeError):
                        errors.append(f"Invalid type for {field.name}: expected {field.data_type.__name__}")
                
                # Custom validators
                for validator in field.validators:
                    try:
                        if not validator(value):
                            errors.append(f"Validation failed for {field.name}")
                    except Exception as e:
                        errors.append(f"Validator error for {field.name}: {str(e)}")
        
        return len(errors) == 0, errors
    
    def validate_all(self, rows: Iterator[DataRow]) -> ProcessingStats:
        """Validate all rows and return statistics."""
        stats = ProcessingStats()
        
        for row in rows:
            stats.total_records += 1
            is_valid, errors = self.validate(row)
            
            if is_valid:
                stats.valid_records += 1
            else:
                stats.invalid_records += 1
                stats.errors.extend(errors)
        
        return stats
    
    # Dead code: unused method
    def unused_validation_method(self, row: DataRow) -> bool:
        """This method is never called."""
        if not row:
            return False
        for field in self.schema.fields:
            if field.name not in row:
                return False
        return True


class DataTransformer:
    """Transform data records."""
    
    def __init__(self):
        self._transforms: List[Callable[[DataRow], DataRow]] = []
    
    def add_transform(self, func: Callable[[DataRow], DataRow]) -> 'DataTransformer':
        """Add a transformation function."""
        self._transforms.append(func)
        return self
    
    def transform(self, row: DataRow) -> DataRow:
        """Apply all transformations to a row."""
        result = row.copy()
        for func in self._transforms:
            result = func(result)
        return result
    
    def transform_all(self, rows: Iterator[DataRow]) -> Iterator[DataRow]:
        """Transform all rows."""
        for row in rows:
            yield self.transform(row)
    
    @staticmethod
    def lowercase_keys(row: DataRow) -> DataRow:
        """Transform all keys to lowercase."""
        return {k.lower(): v for k, v in row.items()}
    
    @staticmethod
    def trim_strings(row: DataRow) -> DataRow:
        """Trim whitespace from string values."""
        return {k: v.strip() if isinstance(v, str) else v for k, v in row.items()}
    
    @staticmethod
    def remove_nulls(row: DataRow) -> DataRow:
        """Remove null/None values."""
        return {k: v for k, v in row.items() if v is not None}
    
    @staticmethod
    def add_timestamp(row: DataRow) -> DataRow:
        """Add processing timestamp."""
        result = row.copy()
        result['_processed_at'] = datetime.now().isoformat()
        return result


class DataFilter:
    """Filter data records based on conditions."""
    
    def __init__(self):
        self._filters: List[FilterFunc] = []
    
    def add_filter(self, func: FilterFunc) -> 'DataFilter':
        """Add a filter function."""
        self._filters.append(func)
        return self
    
    def matches(self, row: DataRow) -> bool:
        """Check if row matches all filters."""
        return all(func(row) for func in self._filters)
    
    def filter(self, rows: Iterator[DataRow]) -> Iterator[DataRow]:
        """Filter rows."""
        for row in rows:
            if self.matches(row):
                yield row
    
    @staticmethod
    def equals(field: str, value: Any) -> FilterFunc:
        """Create equality filter."""
        return lambda row: row.get(field) == value
    
    @staticmethod
    def greater_than(field: str, value: Any) -> FilterFunc:
        """Create greater-than filter."""
        return lambda row: row.get(field, 0) > value
    
    @staticmethod
    def contains(field: str, substring: str) -> FilterFunc:
        """Create contains filter."""
        return lambda row: substring in str(row.get(field, ''))
    
    @staticmethod
    def regex_match(field: str, pattern: str) -> FilterFunc:
        """Create regex filter."""
        compiled = re.compile(pattern)
        return lambda row: bool(compiled.search(str(row.get(field, ''))))
    
    @staticmethod
    def not_null(field: str) -> FilterFunc:
        """Create not-null filter."""
        return lambda row: row.get(field) is not None


# Dead code: unused class
class DeprecatedAggregator:
    """Old aggregation class - no longer used."""
    
    def __init__(self):
        self.results = {}
    
    def sum(self, field: str, rows: Iterator[DataRow]) -> float:
        """Calculate sum of field values."""
        total = 0.0
        for row in rows:
            if field in row:
                total += float(row[field])
        return total
    
    def count(self, rows: Iterator[DataRow]) -> int:
        """Count rows."""
        return sum(1 for _ in rows)


class DataAggregator:
    """Aggregate data by fields."""
    
    def __init__(self, group_by: List[str]):
        self.group_by = group_by
        self._aggregations: Dict[str, Tuple[str, str]] = {}  # field -> (agg_type, source)
    
    def add_aggregation(self, name: str, agg_type: str, source_field: str) -> 'DataAggregator':
        """Add an aggregation."""
        self._aggregations[name] = (agg_type, source_field)
        return self
    
    def aggregate(self, rows: Iterator[DataRow]) -> Dict[Tuple, DataRow]:
        """Perform aggregation."""
        groups: Dict[Tuple, List[DataRow]] = defaultdict(list)
        
        for row in rows:
            key = tuple(row.get(f) for f in self.group_by)
            groups[key].append(row)
        
        results = {}
        for key, group_rows in groups.items():
            result = dict(zip(self.group_by, key))
            
            for name, (agg_type, source) in self._aggregations.items():
                values = [r.get(source) for r in group_rows if r.get(source) is not None]
                
                if agg_type == 'sum':
                    result[name] = sum(values)
                elif agg_type == 'avg':
                    result[name] = sum(values) / len(values) if values else 0
                elif agg_type == 'count':
                    result[name] = len(values)
                elif agg_type == 'min':
                    result[name] = min(values) if values else None
                elif agg_type == 'max':
                    result[name] = max(values) if values else None
            
            results[key] = result
        
        return results


class DataPipeline:
    """Pipeline for processing data through multiple stages."""
    
    def __init__(self, name: str = 'pipeline'):
        self.name = name
        self._stages: List[Tuple[str, Callable]] = []
        self._stats = ProcessingStats()
    
    def add_stage(self, name: str, func: Callable) -> 'DataPipeline':
        """Add a processing stage."""
        self._stages.append((name, func))
        return self
    
    def run(self, data: Iterator[DataRow]) -> Iterator[DataRow]:
        """Run the pipeline."""
        start_time = datetime.now()
        result = data
        
        for stage_name, func in self._stages:
            logger.debug(f"Running stage: {stage_name}")
            result = func(result)
        
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        self._stats.processing_time_ms = elapsed
        
        return result
    
    def get_stats(self) -> ProcessingStats:
        """Get processing statistics."""
        return self._stats
    
    # Dead code: unused method  
    def deprecated_parallel_run(self, data: List[DataRow]) -> List[DataRow]:
        """Old parallel processing - deprecated."""
        import concurrent.futures
        
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_single, row) for row in data]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return results
    
    def _process_single(self, row: DataRow) -> DataRow:
        """Process a single row through all stages."""
        result = row
        for _, func in self._stages:
            result = list(func(iter([result])))[0]
        return result


class DataCache:
    """Thread-safe cache for data operations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return value
                else:
                    del self._cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            self._cache[key] = (value, datetime.now())
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if self._cache:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]


# Utility functions

def hash_row(row: DataRow) -> str:
    """Generate hash for a data row."""
    content = json.dumps(row, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def merge_rows(row1: DataRow, row2: DataRow, 
               conflict_strategy: str = 'prefer_first') -> DataRow:
    """Merge two data rows."""
    result = row1.copy()
    
    for key, value in row2.items():
        if key not in result:
            result[key] = value
        elif conflict_strategy == 'prefer_second':
            result[key] = value
        elif conflict_strategy == 'combine':
            if isinstance(result[key], list):
                result[key].append(value)
            else:
                result[key] = [result[key], value]
    
    return result


def flatten_nested(row: DataRow, separator: str = '.') -> DataRow:
    """Flatten nested dictionaries."""
    result = {}
    
    def _flatten(obj: Any, prefix: str = ''):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                _flatten(value, new_key)
        else:
            result[prefix] = obj
    
    _flatten(row)
    return result


# Dead code: unused function
def unused_utility_function(data: List[DataRow]) -> Dict[str, int]:
    """This utility function is never called."""
    counts = defaultdict(int)
    for row in data:
        for key in row:
            counts[key] += 1
    return dict(counts)


@lru_cache(maxsize=128)
def parse_date(date_str: str, format: str = '%Y-%m-%d') -> datetime:
    """Parse date string with caching."""
    return datetime.strptime(date_str, format)


def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        import time
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


# ----- Main Entry Point -----

if __name__ == '__main__':
    # Example usage
    data = [
        {'name': 'Alice', 'age': 30, 'city': 'NYC'},
        {'name': 'Bob', 'age': 25, 'city': 'LA'},
        {'name': 'Charlie', 'age': 35, 'city': 'NYC'},
    ]
    
    source = InMemoryDataSource(data)
    
    transformer = DataTransformer()
    transformer.add_transform(DataTransformer.lowercase_keys)
    transformer.add_transform(DataTransformer.add_timestamp)
    
    filter = DataFilter()
    filter.add_filter(DataFilter.greater_than('age', 24))
    
    pipeline = DataPipeline('example_pipeline')
    pipeline.add_stage('transform', transformer.transform_all)
    pipeline.add_stage('filter', filter.filter)
    
    results = list(pipeline.run(source.read()))
    
    print(f"Processed {len(results)} records:")
    for row in results:
        print(f"  {row}")
