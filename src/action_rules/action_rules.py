import pandas as pd
import itertools
from collections import defaultdict
from candidates.candidate_generator import CandidateGenerator
from rules.rules import Rules
from output.output import Output


class ActionRules:
    def __init__(self, min_stable_attributes: int, min_flexible_attributes: int, min_undesired_support: int,
                 min_undesired_confidence: float, min_desired_support: int, min_desired_confidence: float,
                 verbose=False):
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.verbose = verbose
        self.rules = None
        self.output = None

    def fit(self, data: pd.DataFrame, stable_attributes: list, flexible_attributes: list, target: str,
            undesired_state: str, desired_state: str):
        data = pd.get_dummies(data, sparse=False, columns=data.columns, prefix_sep='_<item>_')
        stable_items_binding, flexible_items_binding, target_items_binding = self.get_bindings(data, stable_attributes,
                                                                                               flexible_attributes,
                                                                                               target)
        stop_list = self.get_stop_list(stable_items_binding, flexible_items_binding)
        frames = self.get_split_tables(data, target_items_binding, target)
        undesired_state = target + '_<item>_' + str(undesired_state)
        desired_state = target + '_<item>_' + str(desired_state)

        stop_list_itemset = []

        candidates_queue = [{
            'ar_prefix': tuple(),
            'itemset_prefix': tuple(),
            'stable_items_binding': stable_items_binding,
            'flexible_items_binding': flexible_items_binding,
            'undesired_mask': None,
            'desired_mask': None,
            'actionable_attributes': 0
        }]
        k = 0
        self.rules = Rules(undesired_state, desired_state)
        candidate_generator = CandidateGenerator(frames, self.min_stable_attributes, self.min_flexible_attributes,
                                                 self.min_undesired_support, self.min_desired_support,
                                                 self.min_undesired_confidence, self.min_desired_confidence,
                                                 undesired_state, desired_state, rules)
        while len(candidates_queue) > 0:
            candidate = candidates_queue.pop(0)
            if len(candidate['ar_prefix']) > k:
                k += 1
                self.rules.prune_classification_rules(k, stop_list)
            new_candidates = candidate_generator.generate_candidates(**candidate, stop_list=stop_list,
                                                                     stop_list_itemset=stop_list_itemset,
                                                                     undesired_state=undesired_state,
                                                                     desired_state=desired_state,
                                                                     verbose=self.verbose)
            candidates_queue += new_candidates
        self.rules.generate_action_rules()
        self.output = Output(self.rules.action_rules)

    def get_bindings(self, data, stable_attributes, flexible_attributes, target):
        stable_items_binding = defaultdict(lambda: [])
        flexible_items_binding = defaultdict(lambda: [])
        target_items_binding = defaultdict(lambda: [])

        for col in data.columns:
            is_continue = False
            # stable
            for attribute in stable_attributes:
                if col.startswith(attribute + '_<item>_'):
                    stable_items_binding[attribute].append(col)
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # flexible
            for attribute in flexible_attributes:
                if col.startswith(attribute + '_<item>_'):
                    flexible_items_binding[attribute].append(col)
                    is_continue = True
                    break
            if is_continue is True:
                continue
            # target
            if col.startswith(target + '_<item>_'):
                target_items_binding[target].append(col)
        return stable_items_binding, flexible_items_binding, target_items_binding

    def get_stop_list(self, stable_items_binding, flexible_items_binding):
        stop_list = []
        for items in stable_items_binding.values():
            for stop_couple in itertools.product(items, repeat=2):
                stop_list.append(tuple(stop_couple))
        for item in flexible_items_binding.keys():
            stop_list.append(tuple([item, item]))
        return stop_list

    def get_split_tables(self, data, target_items_binding, target):
        frames = {}
        for item in target_items_binding[target]:
            mask = data[item] == 1
            frames[item] = data[mask]
        return frames
