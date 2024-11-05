# These fields are available in the patterns dataframe and we always want to keep them
BASE_COLUMNS = ["support", "itemset", "length", "support_count"]


def sort_pattern(x, abbreviations={}):
    """Sort a pattern and abbreviate by replacing the abbreviations"""
    x = list(x)
    x.sort()
    x = ", ".join(x)
    for k, v in abbreviations.items():
        x = x.replace(k, v)
    return x


def abbreviate_dictionary(pattern_value, abbreviations):
    # Shapley values (dict) as input
    return {
        frozenset([sort_pattern(k, abbreviations=abbreviations)]): v
        for k, v in pattern_value.items()
    }


class DivergencePatternProcessor:
    def __init__(self, patterns, metric):
        """
        :param patterns: dataframe patterns
        :param metric: metric used for divergence
        """
        # We only want to keep the columns that are relevant to the metric + the base columns
        self.patterns = patterns[
            BASE_COLUMNS + [metric, f"{metric}_div", f"{metric}_t"]
        ]
        self.metric = metric
        self.dict_len_pattern_divergence = (
            self.dict_len_pattern_divergence_representation()
        )

    def dict_len_pattern_divergence_representation(self):
        """
        Define an intermediate representation of the patterns dataframe in the form
        Len itemset -> {pattern: divergence}
        :param metric: metric used for divergence
        """
        patterns = self.patterns
        metric = self.metric + "_div"
        d = patterns[["itemset", metric]].set_index("itemset").to_dict("index")
        return {
            k: {k1: v[metric] for k1, v in d.items() if k == len(k1)}
            for k in range(0, max(patterns["length"] + 1))
        }

    def shapley_value(self, pattern=None, row_idx=None):
        """
        Compute the Shapley value of a pattern
        We can specify the pattern either directly by specifying the pattern (frozen set) or row_idx of the pattern in the patterns dataframe
        Args:
            pattern (frozen set): list of items - if None, row_idx must be provided
            row_idx (int): row index of the pattern in the patterns dataframe - if None, pattern must be provided
        Returns:
            (dict) Shapley value of the pattern - {item: shapley value} for each item in the pattern
        """
        assert (
            pattern is None or row_idx is None
        ), "Either pattern or row_idx must be provided"
        assert (
            pattern is not None or row_idx is not None
        ), "Either pattern or row_idx must be provided"

        if row_idx is not None:
            pattern = self.patterns.iloc[row_idx]["itemset"]
        from divexplorer.shapley_value import compute_shapley_value

        return compute_shapley_value(pattern, self.dict_len_pattern_divergence)

    def plot_shapley_value(
        self,
        pattern : frozenset =None,
        row_idx : int =None,
        shapley_values : dict =None,
        figsize : tuple=(4, 3),
        abbreviations : dict ={},
        sort_by_value: bool =True,
        height : float =0.5,
        linewidth : float=0.8,
        labelsize: int =10,
        title : str="",
        x_label="",
        name_fig : str=None,
        save_fig : bool=False,
        show_figure : bool=True,
    ):
        """
        Plot the Shapley value of a pattern.
        Specify either pattern or row_idx or shapley_value.
        :param pattern: list of items
        :param row_idx: row index of the pattern in the patterns dataframe
        :param shapley_values: dictionary of pattern scores: {pattern: score}
        :param figsize: figure size
        :param abbreviations: dictionary of abbreviations to replace in the patterns - for visualization purposes
        :param sort_by_value: sort the Shapley values by value
        :param height: height of the bars
        :param linewidth: width of the bar border
        :param labelsize: size of the labels
        :param title: title of the plot
        :param x_label: x label
        :param name_fig: name of the figure
        :param save_fig: save the figure
        :param show_figure: show the figure
        """

        assert (
            pattern is None or row_idx is None or shapley_values is None
        ), "Either pattern or row_idx or shapley_value must be provided"

        assert (
            pattern is not None or row_idx is not None or shapley_values is not None
        ), "Either pattern or row_idx or shapley_value must be provided"

        if pattern is not None or row_idx is not None:
            shapley_values = self.shapley_value(pattern=pattern, row_idx=row_idx)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

        if abbreviations:
            shapley_values = abbreviate_dictionary(shapley_values, abbreviations)

        # Dictionary of item in string format and value
        shapleyv_plot = {str(",".join(list(k))): v for k, v in shapley_values.items()}

        if sort_by_value:
            shapleyv_plot = {
                k: v for k, v in sorted(shapleyv_plot.items(), key=lambda item: item[1])
            }

        ax.barh(
            range(len(shapleyv_plot)),
            shapleyv_plot.values(),
            height=height,
            align="center",
            color="#7CBACB",
            linewidth=linewidth,
            edgecolor="#0C4A5B",
        )

        ax.set_yticks(range(len(shapleyv_plot)), minor=False)
        ax.set_yticklabels(list(shapleyv_plot.keys()), minor=False)
        ax.tick_params(axis="y", labelsize=labelsize)
        ax.tick_params(axis="x", labelsize=labelsize)
        ax.set_title(title, fontsize=labelsize)

        ax.set_xlabel(x_label, size=labelsize)

        if save_fig:
            name_fig = "shapley_value.pdf" if name_fig is None else name_fig

            plt.savefig(
                f"{name_fig}",
                bbox_inches="tight",
                pad=0.05,
                facecolor="white",
                transparent=False,
            )
        if show_figure:
            plt.show()
            plt.close()

    def redundancy_pruning(self, th_redundancy):
        """
        Prune the patterns that are redundant with respect to the divergence
        :param th_redundancy: threshold for redundancy
        :returns: a Pandas dataframe containing patterns without redundancy

        Let I and  I - {item i} be two patterns   (for example,  {sex=Male, age=<25} and {sex=Male})
        If exist an item i such that it absolute marginal contribution is lower than a threshold epsilon,
        i.e. abs( divergence(I) - divergence(I - {item i}) <= epsilon
        We can prune I. The pattern 𝐼 - {item i} captures the divergence of pattern 𝐼, since the inclusion of the item i only slightly alters the divergence
        In the example, we would keep just sex=Male
        We proceed in this way for all the patterns.

        """
        patterns_divergence = self.dict_len_pattern_divergence
        redundants = []
        for patterns_len_p in patterns_divergence.values():
            for pattern in patterns_len_p:
                for item in pattern:
                    itemset_minus_item = pattern - frozenset([item])
                    v_itemset_minus_item = patterns_divergence[len(itemset_minus_item)][
                        itemset_minus_item
                    ]

                    v_pattern = patterns_divergence[len(pattern)][pattern]
                    if abs(v_pattern - v_itemset_minus_item) <= th_redundancy:
                        redundants.append(pattern)
        patterns_not_red = self.patterns.loc[
            self.patterns.itemset.isin(redundants) == False
        ]
        return patterns_not_red

    def get_patterns(self, th_redundancy=None, sort_by_divergence=True):
        """
        Return the patterns
        :param th_redundancy: threshold for redundancy - if None, no redundancy pruning
        :param sort_by_divergence: sort the patterns by divergence
        :returns: a Pandas dataframe containing the patterns and their divergence
        """
        if th_redundancy is None:
            patterns = self.patterns
        else:
            patterns = self.redundancy_pruning(th_redundancy)
        if sort_by_divergence:
            return patterns.sort_values(by=[f"{self.metric}_div"], ascending=False)
        else:
            return patterns

    def global_shapley_value(self):
        """Compute the Global Shapley value of the patterns
        The Global Shapley value is a generalization of the Shapley value to the entire set of all items.
        It captures the role of an item in giving rise to divergence jointly with other attributes.
        :returns: A dictionary associating each item to its Global Shapley value.
        """
        # Get 1-itemsets
        items = [item for item in self.dict_len_pattern_divergence[1].keys()]

        # Get attributes of 1-itemsets
        attributes = list(set([list(item)[0].split("=")[0] for item in items]))

        # Get cardinality of attributes

        from collections import Counter

        # Get the cardinality of each attribute - Number of items with that attribute
        cardinality_attributes = dict(
            Counter([list(item)[0].split("=")[0] for item in items])
        )

        global_shapley = {}
        for item in items:
            # Compute Global Shapley value of item
            from divexplorer.shapley_value import global_itemset_divergence

            global_shapley[item] = global_itemset_divergence(
                item,
                self.dict_len_pattern_divergence,
                attributes,
                cardinality_attributes,
            )

        return global_shapley
