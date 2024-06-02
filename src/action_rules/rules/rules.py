from collections import defaultdict


class Rules:
    def __init__(self, undesired_state: str, desired_state: str):
        self.classification_rules = defaultdict(lambda: {'desired': [], 'undesired': []})
        self.undesired_state = undesired_state
        self.desired_state = desired_state
        self.action_rules = []

    def add_classification_rules(self, new_ar_prefix, itemset_prefix, undesired_states, desired_states):
        for undesired_item in undesired_states:
            new_itemset_prefix = itemset_prefix + (undesired_item['item'],)
            self.classification_rules[new_ar_prefix]['undesired'].append({
                'itemset': new_itemset_prefix,
                'support': undesired_item['support'],
                'confidence': undesired_item['confidence'],
                'target': self.undesired_state,
            })
        for desired_item in desired_states:
            new_itemset_prefix = itemset_prefix + (desired_item['item'],)
            self.classification_rules[new_ar_prefix]['desired'].append({
                'itemset': new_itemset_prefix,
                'support': desired_item['support'],
                'confidence': desired_item['confidence'],
                'target': self.desired_state,
            })

    def generate_action_rules(self):
        for attribute_prefix, rules in self.classification_rules.items():
            for desired_rule in rules['desired']:
                for undesired_rule in rules['undesired']:
                    self.action_rules.append({'undesired': undesired_rule, 'desired': desired_rule})

    def prune_classification_rules(self, k, stop_list):
        for attribute_prefix, rules in self.classification_rules.items():
            if k == len(attribute_prefix):
                if len(rules['desired']) < 0 or len(rules['undesired']) < 0:
                    stop_list.append(attribute_prefix)
                    del self.classification_rules[attribute_prefix]
