import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from threading import Lock

# Constants and configuration
class Config:
    VELOCITY_THRESHOLD = 0.5
    FLOW_THEORY_THRESHOLD = 0.8
    INTERACTION_QUALITY_METRICS = ['velocity', 'flow_theory']

class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR

# Exception classes
class AgentException(Exception):
    pass

class InvalidInputException(AgentException):
    pass

class InteractionQualityException(AgentException):
    pass

# Data structures/models
class Interaction:
    def __init__(self, velocity: float, flow_theory: float):
        self.velocity = velocity
        self.flow_theory = flow_theory

class InteractionQuality:
    def __init__(self, interaction: Interaction):
        self.interaction = interaction
        self.quality_metrics = {metric: None for metric in Config.INTERACTION_QUALITY_METRICS}

    def calculate_quality(self):
        self.quality_metrics['velocity'] = self.interaction.velocity > Config.VELOCITY_THRESHOLD
        self.quality_metrics['flow_theory'] = self.interaction.flow_theory > Config.FLOW_THEORY_THRESHOLD

# Validation functions
def validate_interaction(interaction: Interaction) -> bool:
    if interaction.velocity < 0 or interaction.flow_theory < 0:
        return False
    return True

# Utility methods
def calculate_velocity(interaction: Interaction) -> float:
    return interaction.velocity

def calculate_flow_theory(interaction: Interaction) -> float:
    return interaction.flow_theory

# Main class
class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(LogLevel.INFO.value)
        self.lock = Lock()

    def create_interaction(self, velocity: float, flow_theory: float) -> Interaction:
        interaction = Interaction(velocity, flow_theory)
        if not validate_interaction(interaction):
            raise InvalidInputException("Invalid interaction input")
        return interaction

    def calculate_interaction_quality(self, interaction: Interaction) -> InteractionQuality:
        interaction_quality = InteractionQuality(interaction)
        interaction_quality.calculate_quality()
        return interaction_quality

    def log_interaction(self, interaction: Interaction, interaction_quality: InteractionQuality):
        with self.lock:
            self.logger.info(f"Interaction velocity: {interaction.velocity}, flow theory: {interaction.flow_theory}")
            self.logger.info(f"Interaction quality: {interaction_quality.quality_metrics}")

    def process_interaction(self, interaction: Interaction):
        interaction_quality = self.calculate_interaction_quality(interaction)
        self.log_interaction(interaction, interaction_quality)

    def run(self):
        try:
            interaction = self.create_interaction(0.6, 0.9)
            self.process_interaction(interaction)
        except AgentException as e:
            self.logger.error(f"Agent exception: {e}")

# Helper classes and utilities
class InteractionProcessor:
    def __init__(self, agent: Agent):
        self.agent = agent

    def process_interactions(self, interactions: List[Interaction]):
        for interaction in interactions:
            self.agent.process_interaction(interaction)

# Integration interfaces
class AgentInterface:
    @abstractmethod
    def create_interaction(self, velocity: float, flow_theory: float) -> Interaction:
        pass

    @abstractmethod
    def calculate_interaction_quality(self, interaction: Interaction) -> InteractionQuality:
        pass

class AgentFactory:
    @staticmethod
    def create_agent(config: Config) -> Agent:
        return Agent(config)

# Unit test compatibility
class TestAgent:
    def test_create_interaction(self):
        agent = AgentFactory.create_agent(Config())
        interaction = agent.create_interaction(0.6, 0.9)
        assert interaction.velocity == 0.6
        assert interaction.flow_theory == 0.9

    def test_calculate_interaction_quality(self):
        agent = AgentFactory.create_agent(Config())
        interaction = agent.create_interaction(0.6, 0.9)
        interaction_quality = agent.calculate_interaction_quality(interaction)
        assert interaction_quality.quality_metrics['velocity'] is True
        assert interaction_quality.quality_metrics['flow_theory'] is True

if __name__ == "__main__":
    agent = AgentFactory.create_agent(Config())
    agent.run()