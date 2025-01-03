from itertools import chain, combinations
import math


def powerset(iterable):
    s = list(iterable)
    return [
        frozenset(i)
        for i in chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    ]


def compute_shapley_value_item(item, pattern, powerset_pattern, item_score):
    # Get all subsets of the pattern (powerset) where item is present
    subsets_item_i = [s for s in powerset_pattern if item.issubset(s)]

    # Compute the delta score for each subset containing the item
    deltas_item = {}

    for subset in subsets_item_i:
        subset_minus_item = subset - item

        # Delta(s) = v(s) - v(s \ {i})
        # The delta score is the score of the subset containing the item minus the score of the subset without the item
        deltas_item[subset] = (
            item_score[len(subset)][subset]
            - item_score[len(subset_minus_item)][subset_minus_item]
        )

    def weight_delta_score(s, n):
        """Weigh the delta scores
        s = |S| = len(subset) without the item
        n = |N| = len(pattern)
        """
        #  |S!|(|N| - |S| - 1)! / |N|!

        return (math.factorial(s) * math.factorial(n - s - 1)) / math.factorial(n)

    return sum(
        [  # We weight the delta score by |S!|(|N| - |S| - 1)! / |N|! where |S| = len(subset) without the item (len(k) - 1) and |N| = len(pattern)
            weight_delta_score(len(k) - 1, len(pattern)) * v
            for k, v in deltas_item.items()
        ]
    )


def compute_shapley_value(pattern: frozenset, item_score: dict):
    """Compute the Shapley value of a subset of items
    :param pattern: list of items
    :param item_score: dictionary of pattern scores: len(pattern) -> {pattern: score}
    """

    # Get all subsets of the pattern (powerset) - 2^N with N = len(pattern)
    powerset_pattern = powerset(pattern)

    # Initialize the Shapley value dictionary
    shapley_value = {}

    # For each item in the pattern
    for item_i in [frozenset([i]) for i in pattern]:
        # Compute the Shapley value for each item
        shapley_value[item_i] = compute_shapley_value_item(
            item_i, pattern, powerset_pattern, item_score
        )
    return shapley_value


#### Global Shapley Value


import numpy as np


def attribute(item):
    """Returns the attribute of an item.
    Args:
        item (str): item in the form of 'attribute=value'
    Returns:
        str: attribute of the item
    """
    return item.split("=")[0]


def plus(f1, f2):
    return frozenset(list(f1) + list(f2))


def weight_factor_global_shapley_value(lB, lA, lI, prod_mb):
    """Returns the weight factor for the global shapley value."""

    # \frac{|B|! (|A|-|B|-|I|)!} {|A|! \prod_{b \in B \cup attr(I)} m_b}
    import math

    prod_mb = int(prod_mb)

    return (math.factorial(lB) * math.factorial(lA - lB - lI)) / (
        math.factorial(lA) * (prod_mb)
    )


def global_itemset_divergence(I, scores_l, attributes, cardinality_attributes):
    """Returns the global divergence of an itemset.
    Args:
        I (set): itemset
        scores_l (dict): dictionary of len itemset, itemsets and their divergence
        attributes (list): list of attributes (from frequent items)
        card_map (dict): dictionary of attributes and their cardinality, i.e., how many values they can take (from frequent items)
    """
    # Get attributes different from the attribute of the item
    # These are our set B
    Bs = set(attributes) - {attribute(i) for i in I}

    # Get itemsets whose attributes are in B
    # These are our set I_B
    I_Bs = [
        pattern_i
        for len_patterns in scores_l
        for pattern_i in scores_l[len_patterns]
        if [item_i for item_i in pattern_i if attribute(item_i) not in Bs] == []
    ]

    # Initialize
    global_shapley_itemset = 0
    # J \in I_B
    for J in I_Bs:
        JI = plus(J, I)

        if len(JI) in scores_l and JI in scores_l[len(JI)]:
            # Get the attributes in J
            B = [attribute(i) for i in J]

            # Get the attributes in J \cup I
            attr_BI = B + [attribute(i) for i in I]

            prod_mb = np.prod([cardinality_attributes[i] for i in attr_BI])

            w = weight_factor_global_shapley_value(
                len(B), len(attributes), len(I), prod_mb
            )
            # Accumulate the global shapley value
            global_shapley_itemset = (
                global_shapley_itemset
                + (scores_l[len(JI)][JI] - scores_l[len(J)][J]) * w
            )

    return global_shapley_itemset
