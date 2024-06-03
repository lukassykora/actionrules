"""Class Rules."""

from collections import defaultdict


class Rules:
    """
    A class used to manage and generate classification and action rules.

    Attributes
    ----------
    classification_rules : defaultdict
        Default dictionary to store classification rules for undesired and desired states.
    undesired_state : str
        The undesired state of the target attribute.
    desired_state : str
        The desired state of the target attribute.
    action_rules : list
        List to store generated action rules.

    Methods
    -------
    add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states)
        Add classification rules for undesired and desired states.
    generate_action_rules()
        Generate action rules from classification rules.
    prune_classification_rules(k, stop_list)
        Prune classification rules based on their length and update the stop list.
    """

    def __init__(self, undesired_state: str, desired_state: str):
        """
        Initialize the Rules class with the specified undesired and desired states.

        Parameters
        ----------
        undesired_state : str
            The undesired state of the target attribute.
        desired_state : str
            The desired state of the target attribute.
        """
        self.classification_rules = defaultdict(lambda: {'desired': [], 'undesired': []})  # type: defaultdict
        self.undesired_state = undesired_state
        self.desired_state = desired_state
        self.action_rules = []  # type: list

    def add_classification_rules(self, new_ar_prefix, itemset_prefix, undesired_states, desired_states):
        """
        Add classification rules for undesired and desired states.

        Parameters
        ----------
        new_ar_prefix : tuple
            Prefix of the action rule.
        itemset_prefix : tuple
            Prefix of the itemset.
        undesired_states : list
            List of undesired states.
        desired_states : list
            List of desired states.
        """
        for undesired_item in undesired_states:
            new_itemset_prefix = itemset_prefix + (undesired_item['item'],)
            self.classification_rules[new_ar_prefix]['undesired'].append(
                {
                    'itemset': new_itemset_prefix,
                    'support': undesired_item['support'],
                    'confidence': undesired_item['confidence'],
                    'target': self.undesired_state,
                }
            )
        for desired_item in desired_states:
            new_itemset_prefix = itemset_prefix + (desired_item['item'],)
            self.classification_rules[new_ar_prefix]['desired'].append(
                {
                    'itemset': new_itemset_prefix,
                    'support': desired_item['support'],
                    'confidence': desired_item['confidence'],
                    'target': self.desired_state,
                }
            )

    def generate_action_rules(self):
        """Generate action rules from classification rules."""
        for attribute_prefix, rules in self.classification_rules.items():
            for desired_rule in rules['desired']:
                for undesired_rule in rules['undesired']:
                    self.action_rules.append({'undesired': undesired_rule, 'desired': desired_rule})

    def prune_classification_rules(self, k: int, stop_list: list):
        """
        Prune classification rules based on their length and update the stop list.

        Parameters
        ----------
        k : int
            Length of the attribute prefix.
        stop_list : list
            List of prefixes to stop generating rules for.
        """
        for attribute_prefix, rules in self.classification_rules.items():
            if k == len(attribute_prefix):
                if len(rules['desired']) < 0 or len(rules['undesired']) < 0:
                    stop_list.append(attribute_prefix)
                    del self.classification_rules[attribute_prefix]
