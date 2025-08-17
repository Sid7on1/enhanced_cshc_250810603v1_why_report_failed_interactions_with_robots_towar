import logging
import os
import json
from typing import Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = 'config.json'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'Default Agent',
        'type': 'robot'
    },
    'environment': {
        'name': 'Default Environment',
        'type': 'simulated'
    }
}

# Exception classes
class ConfigError(Exception):
    """Base class for configuration-related exceptions."""
    pass

class InvalidConfigError(ConfigError):
    """Raised when the configuration is invalid."""
    pass

class ConfigNotFoundError(ConfigError):
    """Raised when the configuration file is not found."""
    pass

# Data structures/models
@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    type: str

@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    name: str
    type: str

# Configuration class
class Config:
    """Agent and environment configuration."""
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.config = self.load_config()
        self.lock = Lock()

    def load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file '{self.config_file}' not found.")
            raise ConfigNotFoundError(f"Configuration file '{self.config_file}' not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid configuration file: {e}")
            raise InvalidConfigError(f"Invalid configuration file: {e}")

    def save_config(self) -> None:
        """Save configuration to file."""
        with self.lock:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)

    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration."""
        agent_config = self.config.get('agent', DEFAULT_CONFIG['agent'])
        return AgentConfig(**agent_config)

    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment configuration."""
        environment_config = self.config.get('environment', DEFAULT_CONFIG['environment'])
        return EnvironmentConfig(**environment_config)

    def update_agent_config(self, agent_config: AgentConfig) -> None:
        """Update agent configuration."""
        with self.lock:
            self.config['agent'] = agent_config.__dict__
            self.save_config()

    def update_environment_config(self, environment_config: EnvironmentConfig) -> None:
        """Update environment configuration."""
        with self.lock:
            self.config['environment'] = environment_config.__dict__
            self.save_config()

# Helper classes and utilities
class ConfigValidator:
    """Configuration validator."""
    @staticmethod
    def validate_agent_config(agent_config: AgentConfig) -> bool:
        """Validate agent configuration."""
        if not agent_config.name:
            logger.error("Agent name is required.")
            return False
        if not agent_config.type:
            logger.error("Agent type is required.")
            return False
        return True

    @staticmethod
    def validate_environment_config(environment_config: EnvironmentConfig) -> bool:
        """Validate environment configuration."""
        if not environment_config.name:
            logger.error("Environment name is required.")
            return False
        if not environment_config.type:
            logger.error("Environment type is required.")
            return False
        return True

# Main class
class AgentEnvironmentConfig:
    """Agent and environment configuration."""
    def __init__(self, config: Config):
        self.config = config

    def create_agent_config(self, name: str, type: str) -> AgentConfig:
        """Create agent configuration."""
        agent_config = AgentConfig(name, type)
        if not ConfigValidator.validate_agent_config(agent_config):
            raise InvalidConfigError("Invalid agent configuration.")
        return agent_config

    def create_environment_config(self, name: str, type: str) -> EnvironmentConfig:
        """Create environment configuration."""
        environment_config = EnvironmentConfig(name, type)
        if not ConfigValidator.validate_environment_config(environment_config):
            raise InvalidConfigError("Invalid environment configuration.")
        return environment_config

    def update_agent_config(self, agent_config: AgentConfig) -> None:
        """Update agent configuration."""
        self.config.update_agent_config(agent_config)

    def update_environment_config(self, environment_config: EnvironmentConfig) -> None:
        """Update environment configuration."""
        self.config.update_environment_config(environment_config)

    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration."""
        return self.config.get_agent_config()

    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment configuration."""
        return self.config.get_environment_config()

# Unit test compatibility
def create_config() -> Config:
    """Create configuration."""
    return Config()

def create_agent_environment_config(config: Config) -> AgentEnvironmentConfig:
    """Create agent and environment configuration."""
    return AgentEnvironmentConfig(config)

# Example usage
if __name__ == '__main__':
    config = create_config()
    agent_environment_config = create_agent_environment_config(config)

    agent_config = agent_environment_config.create_agent_config('My Agent', 'robot')
    environment_config = agent_environment_config.create_environment_config('My Environment', 'simulated')

    agent_environment_config.update_agent_config(agent_config)
    agent_environment_config.update_environment_config(environment_config)

    print(agent_environment_config.get_agent_config())
    print(agent_environment_config.get_environment_config())