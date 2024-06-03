"""Utilities."""


def calculate_confidence(self, support, opposite_support):
    """
    Calculate the confidence of an action rule.

    Parameters
    ----------
    support : int
        The support value for the desired or undesired state.
    opposite_support : int
        The support value for the opposite state.

    Returns
    -------
    float
        The confidence value calculated as support / (support + opposite_support).
        Returns 0 if the sum of support and opposite_support is 0.
    """
    if support + opposite_support == 0:
        return 0
    return support / (support + opposite_support)
