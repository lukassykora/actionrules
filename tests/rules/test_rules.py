#!/usr/bin/env python
"""Tests for `action_rules` package."""

import pytest

from action_rules.rules.rules import Rules


@pytest.fixture
def rules():
    """Fixture for Rules instance."""
    return Rules('status_<item_target>_default', 'status_<item_target>_paid')


def test_add_classification_rules(rules):
    """Test the add_classification_rules method of Rules."""
    new_ar_prefix = tuple()
    itemset_prefix = tuple()
    undesired_states = [{'item': 'age_<item_stable>_30', 'support': 10, 'confidence': 0.8}]
    desired_states = [{'item': 'age_<item_stable>_30', 'support': 5, 'confidence': 0.6}]
    rules.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
    assert len(rules.classification_rules[new_ar_prefix]['undesired']) > 0
    assert len(rules.classification_rules[new_ar_prefix]['desired']) > 0


def test_generate_action_rules(rules):
    """Test the generate_action_rules method of Rules."""
    new_ar_prefix = tuple()
    itemset_prefix = tuple()
    undesired_states = [{'item': 'age_<item_stable>_30', 'support': 10, 'confidence': 0.8}]
    desired_states = [{'item': 'age_<item_stable>_30', 'support': 5, 'confidence': 0.6}]
    rules.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
    rules.generate_action_rules()
    assert len(rules.action_rules) > 0


def test_prune_classification_rules(rules):
    """Test the prune_classification_rules method of Rules."""
    new_ar_prefix = tuple()
    itemset_prefix = tuple()
    undesired_states = [{'item': 'age_<item_stable>_30', 'support': 10, 'confidence': 0.8}]
    desired_states = []
    rules.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
    stop_list = []
    rules.prune_classification_rules(0, stop_list)
    assert len(stop_list) > 0


def test_calculate_confidence(rules):
    """Test the calculate_confidence method of Rules."""
    confidence = rules.calculate_confidence(10, 5)
    assert confidence == 0.6666666666666666


def test_calculate_uplift(rules):
    """Test the calculate_uplift method of Rules."""
    uplift = rules.calculate_uplift(10, 0.8, 0.6)
    assert uplift == 5.0
