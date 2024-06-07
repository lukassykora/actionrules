#!/usr/bin/env python
"""Tests for `action_rules` package."""

import pandas as pd
import pytest

from action_rules.candidates.candidate_generator import CandidateGenerator
from action_rules.rules.rules import Rules


@pytest.fixture
def sample_frames():
    """Fixture for sample frames to be used in tests."""
    frames = {
        'status_<item_target>_default': pd.DataFrame(
            {
                'age_<item_stable>_30': [1, 0, 0],
                'income_<item_flexible>_low': [1, 0, 0],
                'status_<item_target>_default': [1, 0, 0],
            }
        ),
        'status_<item_target>_paid': pd.DataFrame(
            {
                'age_<item_stable>_30': [0, 1, 0],
                'income_<item_flexible>_low': [0, 1, 0],
                'status_<item_target>_paid': [0, 1, 0],
            }
        ),
    }
    return frames


@pytest.fixture
def rules():
    """Fixture for Rules instance."""
    return Rules('status_<item_target>_default', 'status_<item_target>_paid')


@pytest.fixture
def candidate_generator(sample_frames, rules):
    """Fixture for CandidateGenerator instance."""
    return CandidateGenerator(
        frames=sample_frames,
        min_stable_attributes=1,
        min_flexible_attributes=1,
        min_undesired_support=1,
        min_desired_support=1,
        min_undesired_confidence=0.5,
        min_desired_confidence=0.5,
        undesired_state='status_<item_target>_default',
        desired_state='status_<item_target>_paid',
        rules=rules,
    )


def test_generate_candidates(candidate_generator):
    """Test the generate_candidates method of CandidateGenerator."""
    ar_prefix = tuple()
    itemset_prefix = tuple()
    stable_items_binding = {'age': ['age_<item_stable>_30']}
    flexible_items_binding = {'income': ['income_<item_flexible>_low']}
    undesired_mask = pd.Series([1, 0, 0])
    desired_mask = pd.Series([0, 1, 0])
    actionable_attributes = 0
    stop_list = []
    stop_list_itemset = []
    undesired_state = 'status_<item_target>_default'
    desired_state = 'status_<item_target>_paid'
    new_branches = candidate_generator.generate_candidates(
        ar_prefix,
        itemset_prefix,
        stable_items_binding,
        flexible_items_binding,
        undesired_mask,
        desired_mask,
        actionable_attributes,
        stop_list,
        stop_list_itemset,
        undesired_state,
        desired_state,
        verbose=True,
    )
    assert len(new_branches) > 0


def test_get_frames(candidate_generator):
    """Test the get_frames method of CandidateGenerator."""
    undesired_mask = pd.Series([1, 0, 0])
    desired_mask = pd.Series([0, 1, 0])
    undesired_state = 'status_<item_target>_default'
    desired_state = 'status_<item_target>_paid'
    undesired_frame, desired_frame = candidate_generator.get_frames(
        undesired_mask, desired_mask, undesired_state, desired_state
    )
    assert not undesired_frame.empty
    assert not desired_frame.empty


def test_reduce_candidates_by_min_attributes(candidate_generator):
    """Test the reduce_candidates_by_min_attributes method of CandidateGenerator."""
    stable_items_binding = {'age': ['age_<item_stable>_30', 'age_<item_stable>_40', 'age_<item_stable>_50']}
    flexible_items_binding = {'income': ['income_<item_flexible>_low', 'income_<item_flexible>_high']}
    k = 1
    actionable_attributes = 0
    reduced_stable_items_binding, reduced_flexible_items_binding = (
        candidate_generator.reduce_candidates_by_min_attributes(
            k, actionable_attributes, stable_items_binding, flexible_items_binding
        )
    )
    assert len(reduced_stable_items_binding) == 1
    assert len(reduced_flexible_items_binding) == 0
    k = 2
    actionable_attributes = 0
    reduced_stable_items_binding, reduced_flexible_items_binding = (
        candidate_generator.reduce_candidates_by_min_attributes(
            k, actionable_attributes, stable_items_binding, flexible_items_binding
        )
    )
    assert len(reduced_stable_items_binding) == 1
    assert len(reduced_flexible_items_binding) == 1


def test_process_stable_candidates(candidate_generator):
    """Test the process_stable_candidates method of CandidateGenerator."""
    ar_prefix = tuple()
    itemset_prefix = tuple()
    reduced_stable_items_binding = {'age': ['age_<item_stable>_30']}
    stop_list = []
    stable_candidates = {'age': ['age_<item_stable>_30']}
    undesired_frame = pd.DataFrame({'age_<item_stable>_30': [1, 0, 0]})
    desired_frame = pd.DataFrame({'age_<item_stable>_30': [0, 1, 0]})
    new_branches = []
    candidate_generator.process_stable_candidates(
        ar_prefix,
        itemset_prefix,
        reduced_stable_items_binding,
        stop_list,
        stable_candidates,
        undesired_frame,
        desired_frame,
        new_branches,
        verbose=True,
    )
    assert len(new_branches) > 0


def test_process_flexible_candidates(candidate_generator):
    """Test the process_flexible_candidates method of CandidateGenerator."""
    ar_prefix = tuple()
    itemset_prefix = tuple()
    reduced_flexible_items_binding = {'income': ['income_<item_flexible>_low']}
    stop_list = []
    stop_list_itemset = []
    flexible_candidates = {'income': ['income_<item_flexible>_low']}
    undesired_frame = pd.DataFrame({'income_<item_flexible>_low': [1, 0, 0]})
    desired_frame = pd.DataFrame({'income_<item_flexible>_low': [0, 1, 0]})
    actionable_attributes = 0
    new_branches = []
    candidate_generator.process_flexible_candidates(
        ar_prefix,
        itemset_prefix,
        reduced_flexible_items_binding,
        stop_list,
        stop_list_itemset,
        flexible_candidates,
        undesired_frame,
        desired_frame,
        actionable_attributes,
        new_branches,
        verbose=True,
    )
    assert len(new_branches) > 0


def test_process_items(candidate_generator):
    """Test the process_items method of CandidateGenerator."""
    attribute = 'income'
    items = ['income_<item_flexible>_low']
    itemset_prefix = tuple()
    stop_list_itemset = []
    undesired_frame = pd.DataFrame({'income_<item_flexible>_low': [1, 0, 0]})
    desired_frame = pd.DataFrame({'income_<item_flexible>_low': [0, 1, 0]})
    flexible_candidates = {'income': ['income_<item_flexible>_low']}
    undesired_states, desired_states, undesired_count, desired_count = candidate_generator.process_items(
        attribute,
        items,
        itemset_prefix,
        stop_list_itemset,
        undesired_frame,
        desired_frame,
        flexible_candidates,
        verbose=True,
    )
    assert len(undesired_states) > 0
    assert len(desired_states) > 0


def test_update_new_branches(candidate_generator):
    """Test the update_new_branches method of CandidateGenerator."""
    new_branches = [{'item': 'income_<item_flexible>_low'}]
    stable_candidates = {'age': ['age_<item_stable>_30']}
    flexible_candidates = {'income': ['income_<item_flexible>_low']}
    candidate_generator.update_new_branches(new_branches, stable_candidates, flexible_candidates)
    assert 'stable_items_binding' in new_branches[0]
    assert 'flexible_items_binding' in new_branches[0]
