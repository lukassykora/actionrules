import copy


class CandidateGenerator:

    def __init__(self, frames, min_stable_attributes, min_flexible_attributes,
                 min_undesired_support, min_desired_support,
                 min_undesired_confidence, min_desired_confidence,
                 desired_change_in_target):
        self.frames = frames
        self.min_stable_attributes = min_stable_attributes
        self.min_flexible_attributes = min_flexible_attributes
        self.min_undesired_support = min_undesired_support
        self.min_desired_support = min_desired_support
        self.min_undesired_confidence = min_undesired_confidence
        self.min_desired_confidence = min_desired_confidence
        self.desired_change_in_target = desired_change_in_target

    def generate_candidates(self, ar_prefix, itemset_prefix, stable_items_binding, flexible_items_binding,
                            undesired_mask, desired_mask, actionable_attributes=0, item=0, stop_list=[],
                            undesired_state=0, desired_state=1, stop_list_itemset=[],
                            classification_rules=[], verbose=False):
        K = len(itemset_prefix) + 1
        reduced_stable_items_binding, reduced_flexible_items_binding = self.reduce_candidates_min_attributes(
            K, actionable_attributes, stable_items_binding, flexible_items_binding)

        undesired_frame, desired_frame = self.get_frames(undesired_mask, desired_mask, undesired_state, desired_state)
        stable_candidates = copy.deepcopy(stable_items_binding)
        flexible_candidates = copy.deepcopy(flexible_items_binding)

        new_branches = self.process_stable_candidates(ar_prefix, itemset_prefix, reduced_stable_items_binding,
                                                      stop_list,
                                                      stable_candidates, undesired_frame, desired_frame, verbose)
        self.process_flexible_candidates(ar_prefix, itemset_prefix, reduced_flexible_items_binding, stop_list,
                                         stop_list_itemset, flexible_candidates, undesired_frame, desired_frame,
                                         actionable_attributes, classification_rules, new_branches, verbose)
        self.update_new_branches(new_branches, stable_candidates, flexible_candidates)

        return new_branches

    def reduce_candidates_min_attributes(self, K, actionable_attributes, stable_items_binding, flexible_items_binding):
        # Placeholder for reduce_candidates_min_attributes implementation
        # This should return reduced_stable_items_binding and reduced_flexible_items_binding
        pass

    def get_frames(self, undesired_mask, desired_mask, undesired_state, desired_state):
        if undesired_mask is None:
            return self.frames[undesired_state], self.frames[desired_state]
        else:
            undesired_frame = self.frames[undesired_state].multiply(undesired_mask, axis="index")
            desired_frame = self.frames[desired_state].multiply(desired_mask, axis="index")
            return undesired_frame, desired_frame

    def process_stable_candidates(self, ar_prefix, itemset_prefix, reduced_stable_items_binding, stop_list,
                                  stable_candidates, undesired_frame, desired_frame, verbose):
        new_branches = []

        for attribute, items in reduced_stable_items_binding.items():
            for item in items:
                new_ar_prefix = ar_prefix + (item,)
                if self.in_stop_list(new_ar_prefix, stop_list):
                    continue

                undesired_support = undesired_frame[item].sum()
                desired_support = desired_frame[item].sum()

                if verbose:
                    print('SUPPORT')
                    print(itemset_prefix + (item,))
                    print((undesired_support, desired_support))

                if undesired_support < self.min_undesired_support or desired_support < self.min_desired_support:
                    stable_candidates[attribute].remove(item)
                    stop_list.append(new_ar_prefix)
                else:
                    new_branches.append({'ar_prefix': new_ar_prefix,
                                         'itemset_prefix': new_ar_prefix,
                                         'item': item,
                                         'undesired_mask': undesired_frame[item],
                                         'desired_mask': desired_frame[item],
                                         'actionable_attributes': 0,
                                         })
        return new_branches

    def process_flexible_candidates(self, ar_prefix, itemset_prefix, reduced_flexible_items_binding, stop_list,
                                    stop_list_itemset, flexible_candidates, undesired_frame, desired_frame,
                                    actionable_attributes, classification_rules, new_branches, verbose):
        for attribute, items in reduced_flexible_items_binding.items():
            new_ar_prefix = ar_prefix + (attribute,)
            if self.in_stop_list(new_ar_prefix, stop_list):
                continue

            undesired_states, desired_states, undesired_count, desired_count = self.process_items(
                items, itemset_prefix, stop_list_itemset, undesired_frame, desired_frame, flexible_candidates, verbose)

            if actionable_attributes == 0 and (undesired_count == 0 or desired_count == 0):
                del flexible_candidates[attribute]
                stop_list.append(ar_prefix + (attribute,))
            else:
                for item in items:
                    new_branches.append({'ar_prefix': new_ar_prefix,
                                         'itemset_prefix': itemset_prefix + (item,),
                                         'item': item,
                                         'undesired_mask': undesired_frame[item],
                                         'desired_mask': desired_frame[item],
                                         'actionable_attributes': actionable_attributes + 1,
                                         })
                if actionable_attributes + 1 >= self.min_flexible_attributes:
                    self.add_classification_rules(new_ar_prefix, itemset_prefix, undesired_states, desired_states,
                                                  classification_rules)

    def process_items(self, items, itemset_prefix, stop_list_itemset, undesired_frame, desired_frame,
                      flexible_candidates, verbose):
        undesired_states = []
        desired_states = []
        undesired_count = 0
        desired_count = 0

        for item in items:
            if self.in_stop_list(itemset_prefix + (item,), stop_list_itemset):
                continue

            undesired_support = undesired_frame[item].sum()
            desired_support = desired_frame[item].sum()

            if verbose:
                print('SUPPORT')
                print(itemset_prefix + (item,))
                print((undesired_support, desired_support))

            undesired_conf = self.calculate_confidence(undesired_support, desired_support)
            if undesired_support >= self.min_undesired_support:
                undesired_count += 1
                if undesired_conf >= self.min_undesired_confidence:
                    undesired_states.append({'item': item, 'support': undesired_support, 'confidence': undesired_conf})

            desired_conf = self.calculate_confidence(desired_support, undesired_support)
            if desired_support >= self.min_desired_support:
                desired_count += 1
                if desired_conf >= self.min_desired_confidence:
                    desired_states.append({'item': item, 'support': desired_support, 'confidence': desired_conf})

            if desired_support < self.min_desired_support and undesired_support < self.min_undesired_support:
                flexible_candidates[attribute].remove(item)
                stop_list_itemset.append(itemset_prefix + (item,))

        return undesired_states, desired_states, undesired_count, desired_count

    def calculate_confidence(self, support, opposite_support):
        if support + opposite_support == 0:
            return 0
        return support / (support + opposite_support)

    def add_classification_rules(self, new_ar_prefix, itemset_prefix, undesired_states, desired_states,
                                 classification_rules):
        for undesired_item in undesired_states:
            new_itemset_prefix = itemset_prefix + (undesired_item['item'],)
            classification_rules[new_ar_prefix]['undesired'].append({
                'itemset': new_itemset_prefix,
                'support': undesired_item['support'],
                'confidence': undesired_item['confidence'],
                'target': self.desired_change_in_target[0]
            })
        for desired_item in desired_states:
            new_itemset_prefix = itemset_prefix + (desired_item['item'],)
            classification_rules[new_ar_prefix]['desired'].append({
                'itemset': new_itemset_prefix,
                'support': desired_item['support'],
                'confidence': desired_item['confidence'],
                'target': self.desired_change_in_target[1]
            })

    def update_new_branches(self, new_branches, stable_candidates, flexible_candidates):
        for new_branch in new_branches:
            adding = False
            new_stable = {}
            new_flexible = {}

            for attribute, items in stable_candidates.items():
                for item in items:
                    if adding:
                        if attribute not in new_stable:
                            new_stable[attribute] = []
                        new_stable[attribute].append(item)
                    if item == new_branch['item']:
                        adding = True

            for attribute, items in flexible_candidates.items():
                for item in items:
                    if adding:
                        if attribute not in new_flexible:
                            new_flexible[attribute] = []
                        new_flexible[attribute].append(item)
                    if item == new_branch['item']:
                        adding = True

            new_branch['stable_items_binding'] = new_stable
            new_branch['flexible_items_binding'] = new_flexible

    def in_stop_list(self, item, stop_list):
        # Placeholder for in_stop_list function
        pass
