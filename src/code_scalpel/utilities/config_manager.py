import json
import os
import yaml
from typing import Dict, List, Set, Optional, Union, Any, Type
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import argparse
from collections import defaultdict
import jsonschema
import dotenv
import toml
from functools import lru_cache
import copy

class ConfigSource(Enum):
    """Configuration source types."""
    FILE_JSON = 'json'
    FILE_YAML = 'yaml'
    FILE_TOML = 'toml'
    ENV = 'env'
    CLI = 'cli'
    DEFAULT = 'default'

@dataclass
class ConfigValue:
    """Represents a configuration value with metadata."""
    value: Any
    source: ConfigSource
    schema: Optional[Dict] = None
    description: Optional[str] = None
    required: bool = False
    validators: List[callable] = field(default_factory=list)

class ConfigValidationError(Exception):
    """Exception for configuration validation errors."""
    pass

class ConfigManager:
    """Advanced configuration manager with multiple sources and validation."""
    
    def __init__(self, 
                 config_files: Optional[Union[str, List[str]]] = None,
                 schema_file: Optional[str] = None,
                 env_prefix: str = 'APP_',
                 auto_reload: bool = False):
        self.config_files = config_files if isinstance(config_files, list) else [config_files] if config_files else []
        self.schema_file = schema_file
        self.env_prefix = env_prefix
        self.auto_reload = auto_reload
        
        self.config: Dict[str, ConfigValue] = {}
        self.schema: Optional[Dict] = None
        self.defaults: Dict[str, Any] = {}
        self.modified = False
        
        self._setup_logging()
        self._init_config()

    def _init_config(self):
        """Initialize configuration from all sources."""
        # Load schema if provided
        if self.schema_file:
            self._load_schema()
            
        # Load defaults
        self._load_defaults()
        
        # Load from files
        for file_path in self.config_files:
            self._load_from_file(file_path)
            
        # Load from environment
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with caching.
        
        Args:
            key: Configuration key
            default: Default value if not found
        
        Returns:
            Configuration value
        """
        if self.auto_reload and self.modified:
            self._reload_config()
            
        if key in self.config:
            return self.config[key].value
        return self.defaults.get(key, default)

    def set(self, key: str, value: Any, 
            source: ConfigSource = ConfigSource.DEFAULT):
        """
        Set configuration value with validation.
        
        Args:
            key: Configuration key
            value: Value to set
            source: Source of the configuration
        """
        if self.schema:
            self._validate_value(key, value)
            
        self.config[key] = ConfigValue(
            value=value,
            source=source,
            schema=self.schema.get(key) if self.schema else None
        )
        self.modified = True

    def load_from_dict(self, data: Dict[str, Any],
                      source: ConfigSource = ConfigSource.DEFAULT):
        """Load configuration from dictionary."""
        for key, value in data.items():
            self.set(key, value, source)

    def save_to_file(self, file_path: str,
                    include_metadata: bool = False):
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save configuration
            include_metadata: Whether to include metadata
        """
        data = {}
        
        if include_metadata:
            data = {
                key: {
                    'value': config.value,
                    'source': config.source.value,
                    'description': config.description
                }
                for key, config in self.config.items()
            }
        else:
            data = {
                key: config.value
                for key, config in self.config.items()
            }
            
        extension = Path(file_path).suffix.lower()
        
        if extension == '.json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        elif extension == '.yaml':
            with open(file_path, 'w') as f:
                yaml.dump(data, f)
        elif extension == '.toml':
            with open(file_path, 'w') as f:
                toml.dump(data, f)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def validate(self) -> List[str]:
        """
        Validate entire configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Schema validation
        if self.schema:
            for key, config in self.config.items():
                try:
                    self._validate_value(key, config.value)
                except ConfigValidationError as e:
                    errors.append(str(e))
        
        # Required fields validation
        for key, schema in self.schema.items() if self.schema else {}:
            if schema.get('required', False) and key not in self.config:
                errors.append(f"Missing required configuration: {key}")
        
        # Custom validators
        for key, config in self.config.items():
            for validator in config.validators:
                try:
                    validator(config.value)
                except Exception as e:
                    errors.append(f"Validation error for {key}: {str(e)}")
        
        return errors

    def add_validator(self, key: str, validator: callable):
        """Add custom validator for a configuration key."""
        if key in self.config:
            self.config[key].validators.append(validator)

    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a configuration value."""
        if key in self.config:
            config = self.config[key]
            return {
                'value': config.value,
                'source': config.source.value,
                'schema': config.schema,
                'description': config.description,
                'required': config.required
            }
        return None

    def _load_schema(self):
        """Load JSON schema for configuration validation."""
        try:
            with open(self.schema_file) as f:
                extension = Path(self.schema_file).suffix.lower()
                if extension == '.json':
                    self.schema = json.load(f)
                elif extension == '.yaml':
                    self.schema = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported schema format: {extension}")
        except Exception as e:
            self.logger.error(f"Error loading schema: {str(e)}")
            raise

    def _load_from_file(self, file_path: str):
        """Load configuration from file."""
        if not file_path or not os.path.exists(file_path):
            return
            
        try:
            with open(file_path) as f:
                extension = Path(file_path).suffix.lower()
                
                if extension == '.json':
                    data = json.load(f)
                    source = ConfigSource.FILE_JSON
                elif extension == '.yaml':
                    data = yaml.safe_load(f)
                    source = ConfigSource.FILE_YAML
                elif extension == '.toml':
                    data = toml.load(f)
                    source = ConfigSource.FILE_TOML
                else:
                    raise ValueError(f"Unsupported file format: {extension}")
                    
                self.load_from_dict(data, source)
                
        except Exception as e:
            self.logger.error(f"Error loading config from {file_path}: {str(e)}")
            raise

    def _load_from_env(self):
        """Load configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower()
                try:
                    # Try to parse as JSON for complex values
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
                    
                self.set(config_key, parsed_value, ConfigSource.ENV)

    def _load_defaults(self):
        """Load default configuration values."""
        self.defaults = {
            'log_level': 'INFO',
            'cache_enabled': True,
            'max_retries': 3,
            'timeout': 30,
            'debug': False
        }

    def _validate_value(self, key: str, value: Any):
        """Validate a single configuration value."""
        if not self.schema or key not in self.schema:
            return
            
        schema = self.schema[key]
        
        try:
            # JSON Schema validation
            jsonschema.validate(
                instance={'value': value},
                schema={'properties': {'value': schema}}
            )
            
            # Type validation
            expected_type = schema.get('type')
            if expected_type:
                self._validate_type(value, expected_type)
                
            # Range validation
            if 'minimum' in schema:
                if value < schema['minimum']:
                    raise ConfigValidationError(
                        f"Value for {key} must be >= {schema['minimum']}"
                    )
            if 'maximum' in schema:
                if value > schema['maximum']:
                    raise ConfigValidationError(
                        f"Value for {key} must be <= {schema['maximum']}"
                    )
                    
            # Enum validation
            if 'enum' in schema:
                if value not in schema['enum']:
                    raise ConfigValidationError(
                        f"Value for {key} must be one of {schema['enum']}"
                    )
                    
        except jsonschema.exceptions.ValidationError as e:
            raise ConfigValidationError(f"Validation error for {key}: {str(e)}")

    def _validate_type(self, value: Any, expected_type: str):
        """Validate type of a value."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        if expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                raise ConfigValidationError(
                    f"Expected type {expected_type}, got {type(value)}"
                )

    def _reload_config(self):
        """Reload configuration if modified."""
        if self.modified:
            self._init_config()
            self.modified = False

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ConfigManager')

def create_config_manager(config_files: Optional[Union[str, List[str]]] = None,
                        schema_file: Optional[str] = None) -> ConfigManager:
    """Create a new configuration manager instance."""
    return ConfigManager(config_files, schema_file)