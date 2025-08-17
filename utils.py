import logging
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Tuple

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UtilityFunctions:
    """
    A class containing utility functions for the agent project.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the utility functions with a configuration dictionary.

        Args:
        config (Dict[str, Any]): A dictionary containing configuration settings.
        """
        self.config = config

    def validate_input(self, data: Any) -> bool:
        """
        Validate the input data.

        Args:
        data (Any): The input data to be validated.

        Returns:
        bool: True if the input data is valid, False otherwise.
        """
        try:
            if data is None:
                logger.error("Input data is None")
                return False
            if isinstance(data, (int, float, str, list, dict, tuple)):
                return True
            else:
                logger.error("Invalid input data type")
                return False
        except Exception as e:
            logger.error(f"Error validating input data: {str(e)}")
            return False

    def calculate_velocity(self, data: List[float]) -> float:
        """
        Calculate the velocity using the velocity-threshold algorithm.

        Args:
        data (List[float]): A list of float values.

        Returns:
        float: The calculated velocity.
        """
        try:
            if not self.validate_input(data):
                return 0.0
            if len(data) < 2:
                logger.error("Insufficient data to calculate velocity")
                return 0.0
            velocity = (data[-1] - data[-2]) / (1.0)
            return velocity
        except Exception as e:
            logger.error(f"Error calculating velocity: {str(e)}")
            return 0.0

    def apply_flow_theory(self, data: List[float]) -> float:
        """
        Apply the Flow Theory algorithm to the input data.

        Args:
        data (List[float]): A list of float values.

        Returns:
        float: The result of applying the Flow Theory algorithm.
        """
        try:
            if not self.validate_input(data):
                return 0.0
            if len(data) < 2:
                logger.error("Insufficient data to apply Flow Theory")
                return 0.0
            flow = (data[-1] + data[-2]) / (2.0)
            return flow
        except Exception as e:
            logger.error(f"Error applying Flow Theory: {str(e)}")
            return 0.0

    def calculate_metrics(self, data: List[float]) -> Dict[str, float]:
        """
        Calculate various metrics from the input data.

        Args:
        data (List[float]): A list of float values.

        Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
        """
        try:
            if not self.validate_input(data):
                return {}
            metrics = {}
            metrics['mean'] = np.mean(data)
            metrics['stddev'] = np.std(data)
            metrics['velocity'] = self.calculate_velocity(data)
            metrics['flow'] = self.apply_flow_theory(data)
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}

    def save_data(self, data: Any, filename: str) -> bool:
        """
        Save the input data to a file.

        Args:
        data (Any): The data to be saved.
        filename (str): The filename to save the data to.

        Returns:
        bool: True if the data was saved successfully, False otherwise.
        """
        try:
            if not self.validate_input(data):
                return False
            if not self.validate_input(filename):
                return False
            with open(filename, 'w') as file:
                file.write(str(data))
            return True
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

    def load_data(self, filename: str) -> Any:
        """
        Load data from a file.

        Args:
        filename (str): The filename to load the data from.

        Returns:
        Any: The loaded data.
        """
        try:
            if not self.validate_input(filename):
                return None
            with open(filename, 'r') as file:
                data = file.read()
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

class Configuration:
    """
    A class representing the configuration settings.
    """

    def __init__(self, settings: Dict[str, Any]):
        """
        Initialize the configuration settings.

        Args:
        settings (Dict[str, Any]): A dictionary containing configuration settings.
        """
        self.settings = settings

    def get_setting(self, key: str) -> Any:
        """
        Get a configuration setting by key.

        Args:
        key (str): The key of the setting to retrieve.

        Returns:
        Any: The value of the setting.
        """
        try:
            return self.settings[key]
        except Exception as e:
            logger.error(f"Error getting setting: {str(e)}")
            return None

class ExceptionClasses:
    """
    A class containing custom exception classes.
    """

    class InvalidInputError(Exception):
        """
        An exception class for invalid input errors.
        """

        def __init__(self, message: str):
            """
            Initialize the exception with a message.

            Args:
            message (str): The error message.
            """
            self.message = message
            super().__init__(self.message)

    class DataNotFoundError(Exception):
        """
        An exception class for data not found errors.
        """

        def __init__(self, message: str):
            """
            Initialize the exception with a message.

            Args:
            message (str): The error message.
            """
            self.message = message
            super().__init__(self.message)

def main():
    # Create a configuration object
    config = Configuration({'setting1': 'value1', 'setting2': 'value2'})

    # Create a utility functions object
    utility_functions = UtilityFunctions(config.settings)

    # Test the utility functions
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    metrics = utility_functions.calculate_metrics(data)
    logger.info(f"Metrics: {metrics}")

    # Test the save and load data functions
    filename = 'data.txt'
    utility_functions.save_data(data, filename)
    loaded_data = utility_functions.load_data(filename)
    logger.info(f"Loaded data: {loaded_data}")

if __name__ == '__main__':
    main()