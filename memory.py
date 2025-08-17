import logging
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple, Dict
from collections import deque
from threading import Lock

# Define constants
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
ALPHA = 0.6
BETA = 0.4
EPSILON = 0.001

# Define a custom exception class
class MemoryException(Exception):
    """Custom exception class for memory-related errors."""
    pass

# Define a data structure to store experiences
class Experience:
    """Data structure to store experiences."""
    def __init__(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Initialize an experience.

        Args:
        - state (np.ndarray): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.ndarray): The next state.
        - done (bool): Whether the episode is done.
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

# Define a priority queue data structure
class PriorityReplayBuffer:
    """Priority replay buffer data structure."""
    def __init__(self, capacity: int):
        """
        Initialize a priority replay buffer.

        Args:
        - capacity (int): The maximum capacity of the buffer.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.lock = Lock()

    def add(self, experience: Experience, priority: float):
        """
        Add an experience to the buffer.

        Args:
        - experience (Experience): The experience to add.
        - priority (float): The priority of the experience.
        """
        with self.lock:
            self.buffer.append(experience)
            self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences from the buffer.

        Args:
        - batch_size (int): The size of the batch.

        Returns:
        - List[Experience]: A list of sampled experiences.
        """
        with self.lock:
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            experiences = [self.buffer[i] for i in indices]
            return experiences

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update the priorities of experiences in the buffer.

        Args:
        - indices (List[int]): The indices of the experiences to update.
        - priorities (List[float]): The new priorities of the experiences.
        """
        with self.lock:
            for i, priority in zip(indices, priorities):
                self.priorities[i] = priority

# Define the main memory class
class Memory:
    """Main memory class."""
    def __init__(self, capacity: int = MEMORY_SIZE):
        """
        Initialize the memory.

        Args:
        - capacity (int): The maximum capacity of the memory.
        """
        self.capacity = capacity
        self.buffer = PriorityReplayBuffer(capacity)
        self.logger = logging.getLogger(__name__)

    def add_experience(self, experience: Experience, priority: float):
        """
        Add an experience to the memory.

        Args:
        - experience (Experience): The experience to add.
        - priority (float): The priority of the experience.
        """
        try:
            self.buffer.add(experience, priority)
        except Exception as e:
            self.logger.error(f"Error adding experience: {e}")

    def sample_experiences(self, batch_size: int = BATCH_SIZE) -> List[Experience]:
        """
        Sample a batch of experiences from the memory.

        Args:
        - batch_size (int): The size of the batch.

        Returns:
        - List[Experience]: A list of sampled experiences.
        """
        try:
            experiences = self.buffer.sample(batch_size)
            return experiences
        except Exception as e:
            self.logger.error(f"Error sampling experiences: {e}")
            return []

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update the priorities of experiences in the memory.

        Args:
        - indices (List[int]): The indices of the experiences to update.
        - priorities (List[float]): The new priorities of the experiences.
        """
        try:
            self.buffer.update_priorities(indices, priorities)
        except Exception as e:
            self.logger.error(f"Error updating priorities: {e}")

    def calculate_priority(self, error: float) -> float:
        """
        Calculate the priority of an experience based on the error.

        Args:
        - error (float): The error of the experience.

        Returns:
        - float: The priority of the experience.
        """
        try:
            priority = (error + EPSILON) ** ALPHA
            return priority
        except Exception as e:
            self.logger.error(f"Error calculating priority: {e}")
            return 0.0

# Define a utility function to calculate the TD error
def calculate_td_error(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, q_values: Dict[int, float]) -> float:
    """
    Calculate the TD error.

    Args:
    - state (np.ndarray): The current state.
    - action (int): The action taken.
    - reward (float): The reward received.
    - next_state (np.ndarray): The next state.
    - done (bool): Whether the episode is done.
    - q_values (Dict[int, float]): The Q values.

    Returns:
    - float: The TD error.
    """
    try:
        q_value = q_values[action]
        next_q_value = max(q_values.values())
        td_error = reward + GAMMA * next_q_value * (1 - done) - q_value
        return td_error
    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating TD error: {e}")
        return 0.0

# Define a utility function to calculate the Flow Theory metrics
def calculate_flow_theory_metrics(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
    """
    Calculate the Flow Theory metrics.

    Args:
    - state (np.ndarray): The current state.
    - action (int): The action taken.
    - reward (float): The reward received.
    - next_state (np.ndarray): The next state.
    - done (bool): Whether the episode is done.

    Returns:
    - Dict[str, float]: A dictionary of Flow Theory metrics.
    """
    try:
        metrics = {}
        # Calculate the challenge-skill ratio
        challenge_skill_ratio = np.mean(next_state) / np.mean(state)
        metrics["challenge_skill_ratio"] = challenge_skill_ratio
        # Calculate the flow state
        flow_state = np.mean(next_state) - np.mean(state)
        metrics["flow_state"] = flow_state
        return metrics
    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating Flow Theory metrics: {e}")
        return {}

# Define a utility function to calculate the velocity-threshold metrics
def calculate_velocity_threshold_metrics(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> Dict[str, float]:
    """
    Calculate the velocity-threshold metrics.

    Args:
    - state (np.ndarray): The current state.
    - action (int): The action taken.
    - reward (float): The reward received.
    - next_state (np.ndarray): The next state.
    - done (bool): Whether the episode is done.

    Returns:
    - Dict[str, float]: A dictionary of velocity-threshold metrics.
    """
    try:
        metrics = {}
        # Calculate the velocity
        velocity = np.mean(next_state) - np.mean(state)
        metrics["velocity"] = velocity
        # Calculate the threshold
        threshold = np.mean(next_state) + np.mean(state)
        metrics["threshold"] = threshold
        return metrics
    except Exception as e:
        logging.getLogger(__name__).error(f"Error calculating velocity-threshold metrics: {e}")
        return {}