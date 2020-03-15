import unittest
import pandas as pd

from actionrules.actionRules import ActionRules
from actionrules.desiredState import DesiredState


class TestActionRules(unittest.TestCase):
    def setUp(self):
        self.actionRulesDiscoveryEmptyNotNan = ActionRules([pd.DataFrame()],
                                                           [pd.DataFrame()],
                                                           [pd.DataFrame()],
                                                           DesiredState(),
                                                           [pd.Series()],
                                                           [pd.Series()])
        self.actionRulesDiscoveryEmptyNan = ActionRules([pd.DataFrame()],
                                                        [pd.DataFrame()],
                                                        [pd.DataFrame()],
                                                        DesiredState(),
                                                        [pd.Series()],
                                                        [pd.Series()],
                                                        True)

    def test_is_action_couple_when_stable_candidate_not_nan_same_values(self):
        result = self.actionRulesDiscoveryEmptyNotNan._is_action_couple('0', '0', "stable")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (True, ('0',), False)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_not_stable_candidate_not_nan_different_values(self):
        result = self.actionRulesDiscoveryEmptyNotNan._is_action_couple('0', '1', "stable")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (False, None, True)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_not_stable_candidate_not_nan_missing_value(self):
        result = self.actionRulesDiscoveryEmptyNotNan._is_action_couple('nan', '1', "stable")
        # (bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (False, None, True)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_not_flexible_candidate_not_nan_same_values(self):
        result = self.actionRulesDiscoveryEmptyNotNan._is_action_couple('0', '0', "flexible")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (False, None, True)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_flexible_candidate_not_nan_different_values(self):
        result = self.actionRulesDiscoveryEmptyNotNan._is_action_couple('0', '1', "flexible")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (True, ('0', '1'), False)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_not_flexible_candidate_not_nan_missing_value(self):
        result = self.actionRulesDiscoveryEmptyNotNan._is_action_couple('nan', '1', "flexible")
        # (bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (False, None, True)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_stable_candidate_nan_same_values(self):
        result = self.actionRulesDiscoveryEmptyNan._is_action_couple('0', '0', "stable")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (True, ('0',), False)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_not_stable_candidate_nan_different_values(self):
        result = self.actionRulesDiscoveryEmptyNan._is_action_couple('0', '1', "stable")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (False, None, True)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_stable_candidate_nan_missing_value(self):
        result = self.actionRulesDiscoveryEmptyNan._is_action_couple('nan', '1', "stable")
        # (bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (True, ('1*',), False)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_not_flexible_candidate_nan_same_values(self):
        result = self.actionRulesDiscoveryEmptyNan._is_action_couple('0', '0', "flexible")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (False, None, True)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_flexible_candidate_nan_different_values(self):
        result = self.actionRulesDiscoveryEmptyNan._is_action_couple('0', '1', "flexible")
        #(bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (True, ('0', '1'), False)
        self.assertEqual(expected, result)

    def test_is_action_couple_when_flexible_candidate_nan_missing_value(self):
        result = self.actionRulesDiscoveryEmptyNan._is_action_couple('nan', '1', "flexible")
        # (bool is_action_pair, (before, after) action_pair, bool break_rule)
        expected = (True, ('None', '1'), False)
        self.assertEqual(expected, result)

    def test_get_uplift(self):
        result = self.actionRulesDiscoveryEmptyNan._get_uplift(0.2, 0.8, 0.8)
        expected = 0.15
        self.assertAlmostEqual(expected, result)

if __name__ == '__main__':
    unittest.main()
