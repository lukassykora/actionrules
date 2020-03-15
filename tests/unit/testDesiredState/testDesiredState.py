import unittest

from actionrules.desiredState import DesiredState


class TestDesiredState(unittest.TestCase):
    def setUp(self):
        self.desiredStateDesiredClass = DesiredState(desired_classes = ['1'])
        self.desiredStateDesiredChange = DesiredState(desired_changes = [['0', '1']])

    def test_is_candidate_decision_when_class_true(self):
        result = self.desiredStateDesiredClass.is_candidate_decision('0', '1')
        expected = True
        self.assertEqual(expected, result)

    def test_is_candidate_decision_when_class_false(self):
        result = self.desiredStateDesiredClass.is_candidate_decision('0', '2')
        expected = False
        self.assertEqual(expected, result)

    def test_is_candidate_decision_when_change_true(self):
        result = self.desiredStateDesiredChange.is_candidate_decision('0', '1')
        expected = True
        self.assertEqual(expected, result)

    def test_is_candidate_decision_when_change_false(self):
        result = self.desiredStateDesiredChange.is_candidate_decision('2', '1')
        expected = False
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
