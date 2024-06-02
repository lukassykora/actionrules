def calculate_confidence(self, support, opposite_support):
    if support + opposite_support == 0:
        return 0
    return support / (support + opposite_support)
