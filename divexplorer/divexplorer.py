import numpy as np
import pandas as pd

def get_df_one_hot_encoding(df_input):
    # One-hot encoding of the dataframe
    attributes = df_input.columns
    X_one_hot = pd.get_dummies(df_input, prefix_sep="=", columns=attributes)
    X_one_hot.reset_index(drop=True, inplace=True)
    return X_one_hot


def get_t_value_bayesian(positives, negatives, index_all_data):
    pos_plus_one = 1 + positives
    neg_plus_one = 1 + negatives

    # Mean beta distribution
    mean_beta = pos_plus_one / (pos_plus_one + neg_plus_one)

    # Variance beta distribution
    variance_beta = (pos_plus_one * neg_plus_one) / (
        (pos_plus_one + neg_plus_one) ** 2 * (pos_plus_one + neg_plus_one + 1)
    )

    mean_dataset, var_dataset = mean_beta[index_all_data], variance_beta[index_all_data]
    t_test_value = (abs(mean_beta - mean_dataset)) / (
        (variance_beta + var_dataset) ** 0.5
    )

    return t_test_value


def compute_t_value_bayesian(pos1, tot1, pos2, tot2):
    """Same as above, but meant to be computed for instances, not pandas dataframes."""
    neg1 = tot1 - pos1
    neg2 = tot2 - pos2
    mean_beta_1 = (pos1 + 1) / (tot1 + 2)
    mean_beta_2 = (pos2 + 1) / (tot2 + 2)
    var_beta_1 = (pos1 + 1) * (neg1 + 1) / ((tot1 + 2) ** 2 * (tot1 + 3))
    var_beta_2 = (pos2 + 1) * (neg2 + 1) / ((tot2 + 2) ** 2 * (tot2 + 3))
    t = (abs(mean_beta_1 - mean_beta_2)) / ((var_beta_1 + var_beta_2) ** 0.5)
    return t


def get_welch_t_test(
    squared_values, support_count_values, mean_values, index_all_data
):
    variance_values = squared_values / support_count_values - mean_values**2
    mean_dataset = mean_values[index_all_data]
    variance_dataset = variance_values[index_all_data]
    support_count_dataset = support_count_values[index_all_data]

    # Welch's t-test
    t_test_welch = (abs(mean_values - mean_dataset)) / (
        (
            variance_values / support_count_values
            + variance_dataset / support_count_dataset
        )
        ** 0.5
    )
    return t_test_welch


def compute_t_value_continuos(n1, tot1, tot_sq1, n2, tot2, tot_sq2):
    """Computes the value of Welch's t-test for two continuous variables."""
    if n1 < 2 or n2 < 2:
        return 0 # Not significant. 
    # These are the means. 
    mean1 = tot1 / n1
    mean2 = tot2 / n2
    # These are the population variances. 
    var1 = ((tot_sq1 / n1) - mean1 ** 2) * n1 / (n1 - 1)
    var2 = ((tot_sq2 / n2) - mean2 ** 2) * n2 / (n2 - 1)
    t = (abs(mean1 - mean2)) / ((var1 / n1 + var2 / n2) ** 0.5)
    return t


class DivergenceExplorer:
    def __init__(
        self,
        df,
        is_one_hot_encoding=False,
    ):
        """
        :param df: pandas dataframe.  The columns that one wishes to analyze with divexplorer should have discrete values. 
        :param is_one_hot_encoding: boolean. If True, the dataframe attributes that one wishes to analyze are already one-hot encoded.
            This is useful only for the fpgrowth algorithm; the apriori one can deal with non-one-hot encoded attributes.
        """
        # df_discrete: pandas dataframe with discrete values
        self.df = df

        # is_one_hot_encoding: boolean, if True, the dataframe attributes are already one-hot encoded
        self.is_one_hot_encoding = is_one_hot_encoding

    def get_pattern_divergence(
        self,
        min_support: float,
        max_length: int = None,
        max_instances: int = None,
        seed: int = 1,
        boolean_outcomes: list = None,
        quantitative_outcomes: list = None,
        attributes: list = None,
        FPM_algorithm="fpgrowth",
        show_coincise=True,
    ):
        """
        Computes the divergence of the specified outcomes.  One can specify two types of outcomes: boolean and quantitative.
        The difference lies mainly in the way in which the statistical significance is computed: in both cases, we use
        the Welch's t-test, but for boolean outcomes, we consider the outcomes as Bernoulli random variables. 
        One can specify multiple outcomes simultaneously, as a way to speed up the computation when multiple divergences are needed 
        (compared to computing them one by one).
        :param min_support: minimum support value for the pattern
        :param max_length: maximum length of the patterns
        :param max_instances: maximum number of instances that are read for each itemset. 
        :param boolean_outcomes: list of boolean outcomes
        :param quantitative_outcomes: list of quantitative outcomes
        :param attributes: list of attributes to consider. If missing, all attributes except outcomes are considered. 
        :param FPM_algorithm: algorithm to use for frequent pattern mining
        :param show_coincise: if True, the output is more concise, returning only the average, the divergence and the t value.
        :param seed: seed for the random number generator used to shuffle the dataset in the rand_apriori algorithm.
        The parameters max_length and max_instances are used in the apriori algorithm, which is a version of 
        apriori optimized to compute the divergence while conserving memory.  
        max_length controls the maximum numbers of items in an itemset, and thus, the depth of the search
        in the tree.  
        max_instances controls the maximum number of instances that are read from the dataframe for each itemset. 
        For a support of 0.01, choosing 10000 instances means that 10000 rows are read from the dataset for each itemset, 
        and thus, every itemset is supported by at least 100 instances.
        The greater the number of instances, the more accurate the divergence statistics, the more time the algorithm takes. 
        """

        assert FPM_algorithm in [
            "fpgrowth",
            "apriori",
            "alt_apriori"
        ], f"{FPM_algorithm} algorithm is not handled."

        # Sets the default values for lists. 
        quantitative_outcomes = quantitative_outcomes or []
        boolean_outcomes = boolean_outcomes or []
        attributes = attributes or []

        if attributes == []:
            # Get all attributes except outcomes
            attributes = [
                attr
                for attr in list(self.df.columns)
                if attr not in boolean_outcomes + quantitative_outcomes
            ]

        if FPM_algorithm == "alt_apriori":
            assert len(boolean_outcomes) + len(quantitative_outcomes) > 0, "Some outcome must be specified"
            from divexplorer.utils_alt_apriori import Explorer
            e = Explorer(self.df,
                         columns=attributes,
                         target_columns=boolean_outcomes + quantitative_outcomes, 
                         support_threshold=min_support, 
                         depth_limit=max_length, 
                         count_limit=max_instances)
            e.compute_frequent_itemsets(seed=seed)
            """Returns a dataset, with the following columns:
            - itemset (the itemset, as a frozen set of items)
            - support (the support of the itemset)
            - count (the count of the itemset)
            - length (the length of the itemset)
            And for each target columns T:
            - T_count (the count of T, excluding NANs)
            - T_avg (the average of T, excluding NANs)
            - T_div (the divergence of T)
            - T_t (the significance of the divergence of T)
            """
            # We produce the dataset via a dictionary mapping column names to their values. 
            # Produces the first row. 
            df_dict = {}
            # First, the column for the entire dataset. 
            df_dict['itemset'] = [frozenset()]
            df_dict['support'] = [e.root.support]
            df_dict['support_count'] = [e.root.count]
            df_dict['length'] = [0]
            for i, c in enumerate(boolean_outcomes + quantitative_outcomes):
                df_dict[f'{c}_count'] = [e.root.count_not_nan[i]]
                df_dict[f'{c}'] = [e.root.totals[i] / e.root.count_not_nan[i]]
                df_dict[f'{c}_div'] = [0] # The divergence is 0 by definition.
                df_dict[f'{c}_t'] = [0] # The t value is 0 by definition.
            # We add data for all other rows. 
            for itemset, node in e.frequent_itemsets:
                if len(itemset) == 0:
                    continue # Skip the root; we included it already. 
                df_dict['itemset'].append(frozenset([e.repr_item(i) for i in itemset]))
                df_dict['support'].append(node.support)
                df_dict['support_count'].append(node.count)
                df_dict['length'].append(len(itemset))
                for i, c in enumerate(boolean_outcomes + quantitative_outcomes):
                    df_dict[f'{c}_count'].append(node.count_not_nan[i])
                    avg = node.totals[i] / node.count_not_nan[i]
                    df_dict[f'{c}'].append(avg)
                    df_dict[f'{c}_div'].append(avg - (e.root.totals[i] / e.root.count_not_nan[i]))
                    # The calculation of the t-value differs for boolean and quantitative outcomes.
                    if c in boolean_outcomes:
                        t = compute_t_value_bayesian(node.count_not_nan[i], node.count, 
                                                 e.root.count_not_nan[i], e.root.count)
                    else:
                        t = compute_t_value_continuos(
                            node.count_not_nan[i], node.totals[i], node.totals_squared[i],
                            e.root.count_not_nan[i], e.root.totals[i], e.root.totals_squared[i])
                    df_dict[f'{c}_t'].append(t)
            return pd.DataFrame(df_dict)
    
        
        if max_length is not None and FPM_algorithm == "fpgrowth":
            import warnings

            warnings.warn(
                'The parameter "max_len" is only used for the apriori algorithm. The parameter will be ignored.'
            )

        # Get only the attributes specified
        df_discrete = self.df[attributes]

        len_dataset = len(self.df)

        target_outcomes_names = []

        if self.is_one_hot_encoding == False:
            # If it is not already one-hot encoded, we one-hot encode it
            df_ohe = get_df_one_hot_encoding(df_discrete)

        if quantitative_outcomes:
            # If there are quantitative outcomes, we compute the squared outcome
            df_outcomes = pd.DataFrame()
            for outcome_name in quantitative_outcomes:
                # Compute the squared outcome - we will use it for the divergence computation
                df_outcomes[outcome_name] = self.df[outcome_name].fillna(0)
                df_outcomes[f"{outcome_name}_SQUARED"] = (
                    np.array(df_outcomes[outcome_name].values) ** 2
                )
                df_outcomes[f"{outcome_name}_non_bottom"] = (
                    self.df[outcome_name].isna() == False
                ).astype(int)

                target_outcomes_names.extend(
                    [
                        outcome_name,
                        f"{outcome_name}_SQUARED",
                        f"{outcome_name}_non_bottom",
                    ]
                )
        else:
            # We accumulate the outcomes
            df_outcomes = pd.DataFrame()

            for boolean_outcome in boolean_outcomes:
                positive_outcomes = (self.df[boolean_outcome] == 1).astype(int)
                negative_outcomes = (self.df[boolean_outcome] == 0).astype(int)
                bottom_outcomes = self.df[boolean_outcome].isna().astype(int)
                df_outcomes[f"{boolean_outcome}_positive"] = positive_outcomes
                df_outcomes[f"{boolean_outcome}_negative"] = negative_outcomes
                df_outcomes[f"{boolean_outcome}_bottom"] = bottom_outcomes
                target_outcomes_names.extend(
                    [
                        f"{boolean_outcome}_positive",
                        f"{boolean_outcome}_negative",
                        f"{boolean_outcome}_bottom",
                    ]
                )

        if FPM_algorithm == "fpgrowth":
            from divexplorer.utils_FPgrowth import fpgrowth_cm
            df_outcomes.index = df_ohe.index
            df_divergence = fpgrowth_cm(
                df_ohe.copy(),  # Df with one-hot encoded attributes
                df_outcomes,  # Df with outcomes
                min_support=min_support,  # Minimum support
                columns_accumulate=list(df_outcomes.columns),  # Columns to accumulate
            )
        elif FPM_algorithm == "apriori":
            df_outcomes.index = df_ohe.index
            # We use the apriori algorithm
            df_with_outcomes = pd.concat([df_ohe, df_outcomes], axis=1)

            from divexplorer.utils_apriori import apriori_divergence

            df_divergence = apriori_divergence(
                df_ohe.copy(),
                df_with_outcomes,
                min_support=min_support,
                target_matrix=target_outcomes_names,
                max_len=max_length,
            )

        all_dataset_row = {"support": 1, "itemset": frozenset()}

        cols_to_drop = []
        squared_cols_to_drop = []

        if boolean_outcomes:
            for boolean_outcome in boolean_outcomes:
                # The result is average when non considering the bottom values

                def compute_outcome(positive_outcomes, negative_outcomes):
                    return positive_outcomes / (positive_outcomes + negative_outcomes)

                positive_col_name = f"{boolean_outcome}_positive"
                negative_col_name = f"{boolean_outcome}_negative"
                bottom_col_name = f"{boolean_outcome}_bottom"

                df_divergence[boolean_outcome] = compute_outcome(
                    df_divergence[positive_col_name],
                    df_divergence[negative_col_name],
                )

                cols_to_drop.extend(
                    [positive_col_name, negative_col_name, bottom_col_name]
                )

                # Add the info of the all dataset row
                for column_name in [
                    positive_col_name,
                    negative_col_name,
                    bottom_col_name,
                ]:
                    all_dataset_row[column_name] = df_outcomes[column_name].sum()

                all_dataset_row[f"{boolean_outcome}_div"] = (
                    0  # The divergence is 0 by definition
                )
                all_dataset_row[f"{boolean_outcome}_t"] = (
                    0  # The t value is 0 by definition
                )

                # Compute the average of the all dataset row -- as above
                overall_average = compute_outcome(
                    all_dataset_row[positive_col_name],
                    all_dataset_row[negative_col_name],
                )

                # Add the average to the all dataset row
                all_dataset_row[boolean_outcome] = overall_average

                # Compute the divergence
                df_divergence[f"{boolean_outcome}_div"] = (
                    df_divergence[boolean_outcome] - overall_average
                )

                # Compute the t value

                # Get the positive and negative values
                pos, neg = (
                    df_divergence[positive_col_name].values,
                    df_divergence[negative_col_name].values,
                )

                # We append the pos and neg values of the all dataset row
                pos = np.concatenate([[all_dataset_row[positive_col_name]], pos])
                neg = np.concatenate([[all_dataset_row[negative_col_name]], neg])
                t = get_t_value_bayesian(pos, neg, 0)
                df_divergence[f"{boolean_outcome}_t"] = t[
                    1:
                ]  # We omit the all dataset row

        else:
            for quantitative_outcome in quantitative_outcomes:

                quantitative_outcome_non_bottom = f"{quantitative_outcome}_non_bottom"
                df_divergence[quantitative_outcome] = (
                    df_divergence[quantitative_outcome]
                    / df_divergence[quantitative_outcome_non_bottom]
                )

                quantitative_outcome_squared = f"{quantitative_outcome}_SQUARED"
                squared_cols_to_drop.append(quantitative_outcome_squared)
                squared_cols_to_drop.append(quantitative_outcome_non_bottom)

                # Add the info of the all dataset row
                for column_name in [
                    quantitative_outcome,
                    quantitative_outcome_squared,
                    quantitative_outcome_non_bottom,
                ]:
                    all_dataset_row[column_name] = df_outcomes[column_name].sum()

                all_dataset_row[f"{quantitative_outcome}_div"] = (
                    0  # The divergence is 0 by definition
                )
                all_dataset_row[f"{quantitative_outcome}_t"] = (
                    0  # The t value is 0 by definition
                )

                # Compute the average of the all dataset row -- as above
                overall_average = (
                    all_dataset_row[quantitative_outcome]
                    / all_dataset_row[quantitative_outcome_non_bottom]
                )

                # Add the average to the all dataset row
                all_dataset_row[quantitative_outcome] = overall_average

                # Compute the divergence
                df_divergence[f"{quantitative_outcome}_div"] = (
                    df_divergence[quantitative_outcome] - overall_average
                )

                # Compute the t value with Welch's t-test
                squared_values = df_divergence[quantitative_outcome_squared].values
                # support_count_values = (
                #     df_divergence["support"].values * len_dataset
                # ).round()
                mean_values = df_divergence[quantitative_outcome].values

                # We append the pos and neg values of the all dataset row
                squared_values = np.concatenate(
                    [[all_dataset_row[quantitative_outcome_squared]], squared_values]
                )
                # support_count_values = np.concatenate(
                #     [[len_dataset], support_count_values]
                # )
                mean_values = np.concatenate(
                    [[all_dataset_row[quantitative_outcome]], mean_values]
                )

                non_bottom_values = df_divergence[
                    quantitative_outcome_non_bottom
                ].values
                non_bottom_values = np.concatenate(
                    [
                        [all_dataset_row[quantitative_outcome_non_bottom]],
                        non_bottom_values,
                    ]
                )

                t = get_welch_t_test(
                    squared_values, non_bottom_values, mean_values, 0)
                df_divergence[f"{quantitative_outcome}_t"] = t[
                    1:
                ]  # We omit the all dataset row

        # Add the all dataset row
        df_divergence.loc[len(df_divergence), all_dataset_row.keys()] = (
            all_dataset_row.values()
        )

        df_divergence["length"] = df_divergence["itemset"].str.len()
        df_divergence["support_count"] = (
            df_divergence["support"] * len_dataset
        ).round()
        df_divergence = df_divergence.reset_index(drop=True)

        df_divergence.sort_values("support", ascending=False, inplace=True)
        df_divergence = df_divergence.reset_index(drop=True)

        if show_coincise:
            df_divergence = df_divergence.drop(columns=cols_to_drop)
        df_divergence = df_divergence.drop(columns=squared_cols_to_drop)

        return df_divergence
