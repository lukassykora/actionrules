#!/usr/bin/env python
"""Tests for `action_rules` package."""

import pandas as pd
import pytest

from action_rules.action_rules import ActionRules


@pytest.fixture
def sample_data():
    """Fixture for sample data to be used in tests."""
    data = {
        'age': ['30', '40', '50'],
        'income': ['low', 'medium', 'high'],
        'loan': ['no', 'yes', 'no'],
        'status': ['default', 'paid', 'default'],
    }
    return pd.DataFrame(data)


@pytest.fixture
def action_rules():
    """Fixture for ActionRules instance."""
    return ActionRules(
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_undesired_confidence=0.5,
        min_desired_support=1,
        min_desired_confidence=0.5,
        verbose=True,
    )


def test_fit(action_rules, sample_data):
    """Test the fit method of ActionRules."""
    action_rules.fit(
        sample_data,
        stable_attributes=['age'],
        flexible_attributes=['income', 'loan'],
        target='status',
        undesired_state='default',
        desired_state='paid',
    )
    assert action_rules.rules is not None
    assert action_rules.output is not None


def test_get_bindings(action_rules, sample_data):
    """Test the get_bindings method of ActionRules."""
    stable_attributes = ['age']
    flexible_attributes = ['income']
    target = 'status'
    data = sample_data.astype(str)
    data_stable = pd.get_dummies(data[stable_attributes], sparse=False, prefix_sep='_<item_stable>_')
    data_flexible = pd.get_dummies(data[flexible_attributes], sparse=False, prefix_sep='_<item_flexible>_')
    data_target = pd.get_dummies(data[[target]], sparse=False, prefix_sep='_<item_target>_')
    data = pd.concat([data_stable, data_flexible, data_target], axis=1)
    stable_items_binding, flexible_items_binding, target_items_binding = action_rules.get_bindings(
        data, stable_attributes, flexible_attributes, target
    )
    assert 'age_<item_stable>_30' in stable_items_binding['age']
    assert 'income_<item_flexible>_low' in flexible_items_binding['income']
    assert 'status_<item_target>_default' in target_items_binding['status']


def test_get_stop_list(action_rules, sample_data):
    """Test the get_stop_list method of ActionRules."""
    stable_attributes = ['age']
    flexible_attributes = ['income']
    target = 'status'
    data = sample_data.astype(str)
    data_stable = pd.get_dummies(data[stable_attributes], sparse=False, prefix_sep='_<item_stable>_')
    data_flexible = pd.get_dummies(data[flexible_attributes], sparse=False, prefix_sep='_<item_flexible>_')
    data_target = pd.get_dummies(data[[target]], sparse=False, prefix_sep='_<item_target>_')
    data = pd.concat([data_stable, data_flexible, data_target], axis=1)
    stable_items_binding, flexible_items_binding, target_items_binding = action_rules.get_bindings(
        data, stable_attributes, flexible_attributes, target
    )
    stop_list = action_rules.get_stop_list(stable_items_binding, flexible_items_binding)
    assert ('age_<item_stable>_30', 'age_<item_stable>_30') in stop_list
    assert ('age_<item_stable>_30', 'age_<item_stable>_40') in stop_list
    assert ('age_<item_stable>_40', 'age_<item_stable>_50') in stop_list
    assert ('income', 'income') in stop_list


def test_get_split_tables(action_rules, sample_data):
    """Test the get_split_tables method of ActionRules."""
    stable_attributes = ['age']
    flexible_attributes = ['income']
    target = 'status'
    data = sample_data.astype(str)
    data_stable = pd.get_dummies(data[stable_attributes], sparse=False, prefix_sep='_<item_stable>_')
    data_flexible = pd.get_dummies(data[flexible_attributes], sparse=False, prefix_sep='_<item_flexible>_')
    data_target = pd.get_dummies(data[[target]], sparse=False, prefix_sep='_<item_target>_')
    data = pd.concat([data_stable, data_flexible, data_target], axis=1)
    _, _, target_items_binding = action_rules.get_bindings(data, stable_attributes, flexible_attributes, target)
    frames = action_rules.get_split_tables(data, target_items_binding, target)
    assert 'status_<item_target>_default' in frames
    assert 'status_<item_target>_paid' in frames


def test_get_rules(action_rules, sample_data):
    """Test the get_rules method of ActionRules."""
    action_rules.fit(
        sample_data,
        stable_attributes=['age'],
        flexible_attributes=['income'],
        target='status',
        undesired_state='default',
        desired_state='paid',
    )
    rules = action_rules.get_rules()
    assert rules is not None
