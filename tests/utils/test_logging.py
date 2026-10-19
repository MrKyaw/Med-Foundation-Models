"""Tests for logging utilities."""

import pytest
import torch
import os
from medfm.utils.logging import MetricLogger, SmoothedValue, setup_logging


class TestSmoothedValue:
    def test_update(self):
        sv = SmoothedValue()
        sv.update(1.0)
        sv.update(2.0)
        sv.update(3.0)
        assert sv.count == 3
        assert sv.total == 6.0

    def test_avg(self):
        sv = SmoothedValue()
        sv.update(1.0)
        sv.update(2.0)
        sv.update(3.0)
        assert sv.avg == 2.0

    def test_global_avg(self):
        sv = SmoothedValue()
        sv.update(1.0, n=2)
        sv.update(3.0, n=1)
        assert sv.global_avg == (2.0 + 3.0) / 3

    def test_window_size(self):
        sv = SmoothedValue(window_size=3)
        for i in range(10):
            sv.update(float(i))
        assert len(sv.deque) == 3
        assert sv.max == 9.0

    def test_str(self):
        sv = SmoothedValue()
        sv.update(1.5)
        assert "1.5" in str(sv)


class TestMetricLogger:
    def test_update(self):
        logger = MetricLogger()
        logger.update(loss=0.5, accuracy=0.9)
        assert "loss" in logger.meters
        assert "accuracy" in logger.meters

    def test_str(self):
        logger = MetricLogger()
        logger.update(loss=0.5)
        log_str = str(logger)
        assert "loss" in log_str
        assert "0.5" in log_str

    def test_log_every(self):
        logger = MetricLogger()
        items = list(range(5))
        for i, item in enumerate(logger.log_every(items, print_freq=2, header="Test")):
            pass  # Just ensure it doesn't crash


class TestSetupLogging:
    def test_creates_directory(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        result = setup_logging(log_dir, "test_experiment")
        assert os.path.exists(result)
        assert result != ""

    def test_non_main_process(self, tmp_path):
        result = setup_logging(str(tmp_path), "test_experiment", rank=1)
        assert result == ""
