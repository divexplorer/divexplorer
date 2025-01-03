import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpcommon as fpc
from mlxtend.frequent_patterns.apriori import (
    generate_new_combinations,
    generate_new_combinations_low_memory,
)


def apriori_divergence(
    df,
    df_true_pred,
    min_support=0.5,
    use_colnames=True,
    max_len=None,
    verbose=0,
    low_memory=False,
    target_matrix=["tn", "fp", "fn", "tp"],
    sortedV="support",
):
    """

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemset'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemset' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    """

    def filterColumns(df_filter, cols):
        return df_filter[(df_filter[df_filter.columns[list(cols)]] > 0).all(1)]

    def sum_values(_x):
        out = np.sum(_x, axis=0)
        return np.array(out).reshape(-1)

    def _support(_x, _n_rows, _is_sparse):
        """DRY private method to calculate support as the
        row-wise sum of values / number of rows
        Parameters
        -----------
        _x : matrix of bools or binary
        _n_rows : numeric, number of rows in _x
        _is_sparse : bool True if _x is sparse
        Returns
        -----------
        np.array, shape = (n_rows, )
        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
        """
        out = np.sum(_x, axis=0) / _n_rows
        return np.array(out).reshape(-1)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    fpc.valid_input_check(df)

    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            X = df.values
        else:
            X = df.sparse.to_coo().tocsc()
        is_sparse = True
    else:
        # dense DataFrame
        X = df.values
        is_sparse = False
    support = _support(X, X.shape[0], is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {0: 1, 1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}  # 0: [()],
    conf_metrics = {
        # 0: np.asarray([sum_values(df_true_pred[target_matrix])]),
        1: np.asarray(
            [
                sum_values(filterColumns(df_true_pred, item)[target_matrix])
                for item in itemset_dict[1]
            ]
        ),
    }

    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1

        # With exceptionally large datasets, the matrix operations can use a
        # substantial amount of memory. For low memory applications or large
        # datasets, set `low_memory=True` to use a slower but more memory-
        # efficient implementation.
        if low_memory:
            combin = generate_new_combinations_low_memory(
                itemset_dict[max_itemset], X, min_support, is_sparse
            )
            # slightly faster than creating an array from a list of tuples
            combin = np.fromiter(combin, dtype=int)
            combin = combin.reshape(-1, next_max_itemset + 1)

            if combin.size == 0:
                break
            if verbose:
                print(
                    "\rProcessing %d combinations | Sampling itemset size %d"
                    % (combin.size, next_max_itemset),
                    end="",
                )

            itemset_dict[next_max_itemset] = combin[:, 1:]
            support_dict[next_max_itemset] = combin[:, 0].astype(float) / rows_count
            max_itemset = next_max_itemset
            # TODO
        else:
            combin = generate_new_combinations(itemset_dict[max_itemset])
            combin = np.fromiter(combin, dtype=int)
            combin = combin.reshape(-1, next_max_itemset)

            if combin.size == 0:
                break
            if verbose:
                print(
                    "\rProcessing %d combinations | Sampling itemset size %d"
                    % (combin.size, next_max_itemset),
                    end="",
                )

            if is_sparse:
                _bools = X[:, combin[:, 0]] == all_ones
                for n in range(1, combin.shape[1]):
                    _bools = _bools & (X[:, combin[:, n]] == all_ones)
            else:
                _bools = np.all(X[:, combin], axis=2)
            support = _support(np.array(_bools), rows_count, is_sparse)
            _mask = (support >= min_support).reshape(-1)
            if any(_mask):
                itemset_dict[next_max_itemset] = np.array(combin[_mask])
                support_dict[next_max_itemset] = np.array(support[_mask])
                conf_metrics[next_max_itemset] = np.asarray(
                    [
                        sum_values(filterColumns(df_true_pred, itemset)[target_matrix])
                        for itemset in itemset_dict[next_max_itemset]
                    ]
                )
                max_itemset = next_max_itemset
            else:
                # Exit condition
                break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")
        # conf_matrix_col=pd.Series(list(conf_metrics[k]))
        conf_metrics_cols = pd.DataFrame(list(conf_metrics[k]), columns=target_matrix)

        res = pd.concat((support, itemsets, conf_metrics_cols), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemset"] + target_matrix

    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df["itemset"] = res_df["itemset"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )

    res_df["length"] = res_df["itemset"].str.len()
    res_df["support_count"] = np.sum(res_df[target_matrix], axis=1)

    res_df.sort_values(sortedV, ascending=False, inplace=True)
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df
