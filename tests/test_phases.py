"""Unit tests for Phase 1 and Phase 2"""

import unittest
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig, TrainingConfig, DataConfig, SimplificationConfig
from src.data import DataProcessor
from src.utils import setup_logging, Timer, format_time, count_parameters


class TestConfig(unittest.TestCase):
    """Test configuration classes"""

    def test_model_config(self):
        config = ModelConfig()
        self.assertEqual(config.model_name, "t5-base")
        self.assertEqual(config.max_input_length, 256)
        self.assertEqual(config.max_target_length, 64)

    def test_training_config(self):
        config = TrainingConfig()
        self.assertEqual(config.num_train_epochs, 3)
        self.assertEqual(config.learning_rate, 3e-4)

    def test_data_config(self):
        config = DataConfig()
        self.assertEqual(config.train_split, 0.9)

    def test_simplification_config(self):
        config = SimplificationConfig()
        self.assertIn("beginner", config.vocabulary_levels)
        self.assertIn("intermediate", config.vocabulary_levels)
        self.assertIn("advanced", config.vocabulary_levels)


class TestDataProcessor(unittest.TestCase):
    """Test data processing"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DataProcessor(
            data_dir=self.temp_dir, cache_dir=os.path.join(self.temp_dir, "cache")
        )

    def test_create_sample_data(self):
        data = self.processor.create_sample_data(num_samples=100)
        self.assertEqual(len(data), 100)
        self.assertIn("source", data[0])
        self.assertIn("target", data[0])

    def test_split_data(self):
        raw_data = [
            {"id": i, "source": f"text {i}", "target": f"simple {i}"}
            for i in range(100)
        ]
        train, val = self.processor.split_data(raw_data, train_ratio=0.8)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(val), 20)

    def test_augment_data(self):
        raw_data = [
            {"id": i, "source": f"text {i}", "target": f"simple {i}"} for i in range(10)
        ]
        augmented = self.processor.augment_data(raw_data, augmentation_factor=3)
        self.assertEqual(len(augmented), 30)


class TestUtils(unittest.TestCase):
    """Test utility functions"""

    def test_format_time(self):
        self.assertEqual(format_time(30), "30s")
        self.assertEqual(format_time(90), "1m 30s")
        self.assertEqual(format_time(3661), "1h 1m 1s")

    def test_setup_logging(self):
        logger = setup_logging("test")
        self.assertEqual(logger.name, "test")
        logger.info("Test message")

    def test_timer(self):
        import time

        with Timer("test") as t:
            time.sleep(0.01)


class TestSimplificationConfig(unittest.TestCase):
    """Test simplification configuration"""

    def test_vocabulary_levels(self):
        config = SimplificationConfig()
        beginner_words = config.vocabulary_levels.get("beginner", [])
        self.assertIn("A1", beginner_words)
        self.assertIn("A2", beginner_words)

    def test_default_mode(self):
        config = SimplificationConfig()
        self.assertEqual(config.default_mode, "replace")


if __name__ == "__main__":
    unittest.main()
