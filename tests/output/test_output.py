#!/usr/bin/env python
"""Tests for `action_rules` package."""

import pytest

from action_rules.output.output import Output


@pytest.fixture
def sample_action_rules():
    """Fixture for sample action rules to be used in tests."""
    return [
        {
            'undesired': {
                'itemset': ['age_<item_stable>_30', 'income_<item_flexible>_low'],
                'support': 10,
                'confidence': 0.8,
                'target': 'status_<item_target>_default',
            },
            'desired': {
                'itemset': ['age_<item_stable>_30', 'income_<item_flexible>_medium'],
                'support': 5,
                'confidence': 0.6,
                'target': 'status_<item_target>_paid',
            },
            'uplift': 0.2,
        }
    ]


@pytest.fixture
def output(sample_action_rules):
    """Fixture for Output instance."""
    return Output(sample_action_rules, 'status')


def test_get_ar_notation(output):
    """Test the get_ar_notation method of Output."""
    ar_notation = output.get_ar_notation()
    assert len(ar_notation) > 0
    assert 'age: 30' in ar_notation[0]


def test_get_export_notation(output):
    """Test the get_export_notation method of Output."""
    export_notation = output.get_export_notation()
    assert len(export_notation) > 0
    assert '"attribute": "age"' in export_notation


def test_get_pretty_ar_notation(output):
    """Test the get_pretty_ar_notation method of Output."""
    pretty_ar_notation = output.get_pretty_ar_notation()
    assert len(pretty_ar_notation) > 0
    assert "If attribute 'age' is '30'" in pretty_ar_notation[0]
