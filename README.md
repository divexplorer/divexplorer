[![PyPI](https://img.shields.io/pypi/v/divexplorer)](https://pypi.org/project/divexplorer/)
[![Downloads](https://pepy.tech/badge/divexplorer)](https://pepy.tech/project/divexplorer)

# DivExplorer

Machine learning models may perform differently on different data subgroups. We propose the notion of divergence over itemsets (i.e., conjunctions of simple predicates) as a measure of different classification behavior on data subgroups, and the use of frequent pattern mining techniques for their identification. We quantify the contribution of different attribute values to divergence with the notion of Shapley values to identify both critical and peculiar behaviors of attributes.
See our [paper](https://divexplorer.github.io/static/DivExplorer.pdf) and our [project page](https://divexplorer.github.io/) for all the details.

## Installation

Install using [pip](http://www.pip-installer.org/en/latest) with:

<pre>
pip install divexplorer
</pre>

or, download a wheel or source archive from [PyPI](https://pypi.org/project/divexplorer/).

## Example Notebooks

This [notebook](https://github.com/divexplorer/divexplorer/blob/main/notebooks/DivExplorerExample.ipynb) gives an example of how to use DivExplorer to find divergent subgroups in datasets and in the predictions of a classifier.

## Documentation

For the code details, see the [documentation](https://github.com/divexplorer/divexplorer/blob/main/Documentation.md). 

The original paper is:

> [Looking for Trouble: Analyzing Classifier Behavior via Pattern Divergence](https://divexplorer.github.io/static/DivExplorer.pdf). [Eliana Pastor](https://github.com/elianap), [Luca de Alfaro](https://luca.dealfaro.com/), [Elena Baralis](https://dbdmg.polito.it/wordpress/people/elena-baralis/). In Proceedings of the 2021 ACM SIGMOD Conference, 2021.

You can find more papers and information in the [DivExplorer project page](https://divexplorer.github.io/).


## Quick Start

DivExplorer works on Pandas datasets.  Here we load an example one, and discretize in coarser ranges one of its attributes. 

```python
import pandas as pd

df_census = pd.read_csv('https://raw.githubusercontent.com/divexplorer/divexplorer/main/datasets/census_income.csv')
df_census["AGE_RANGE"] = df_census.apply(lambda row : 10 * (row["A_AGE"] // 10), axis=1)
```

We can then find the data subgroups that have highest income divergence, using the `DivergenceExplorer` class as follows: 

```python
from divexplorer import DivergenceExplorer

fp_diver = DivergenceExplorer(df_census)
subgroups = fp_diver.get_pattern_divergence(
    min_support=0.001,
    attributes=["STATE", "SEX", "EDUCATION", "AGE_RANGE"], 
    quantitative_outcomes=["PTOTVAL"])
subgroups.sort_values(by="PTOTVAL_div", ascending=False).head(10)
```

You can also prune redundant subgroups by specifying:
*  a threshold, so that attributes that don't increase the divergence by at least the threshold value are not included in subgroups, 
* a minimum t-value, to select only significant subgroups.

```python
from divexplorer import DivergencePatternProcessor

processor = DivergencePatternProcessor(subgroups, "PTOTVAL")
pruned_subgroups = pd.DataFrame(processor.redundancy_pruning(th_redundancy=10000))
pruned_subgroups = pruned_subgroups[pruned_subgroups["PTOTVAL_t"] > 2]
pruned_subgroups.sort_values(by="PTOTVAL_div", ascending=False, ignore_index=True)
```

### Finding subgroups with divergent performance in classifiers

For classifiers, it may be of interest to find the subgroups with the highest (or lowest) divergence in characteristics such as false positive rates, etc.  Here is how to do it for the false-positive rate in a COMPAS-derived classifier. 

```python
compas_df = pd.read_csv('https://raw.githubusercontent.com/divexplorer/divexplorer/main/datasets/compas_discretized.csv')
```

We generate an `fp` column whose average will give the false-positive rate, like so: 

```python
from divexplorer.outcomes import get_false_positive_rate_outcome

y_trues = compas_df["class"]
y_preds = compas_df["predicted"]

compas_df['fp'] =  get_false_positive_rate_outcome(y_trues, y_preds)
```

The `fp` column has values: 

* 1, if the data is a false positive (`class` is 0 and `predicted` is 1)
* 0, if the data is a true negative (`class` is 0 and `predicted` is 0). 
* NaN, if the class is positive (`class` is 1).

We use Nan for `class` 1 data, to exclude those data from the average, so that the column average is the false-positive rate.
We can then find the most divergent groups as in the previous example, noting that here we use `boolean_outcomes` rather than `quantitative_outcomes` because `fp` is boolean: 

```python
fp_diver = DivergenceExplorer(compas_df)

attributes = ['race', '#prior', 'sex', 'age']
FP_fm = fp_diver.get_pattern_divergence(min_support=0.1, attributes=attributes, 
                                        boolean_outcomes=['fp'])
FP_fm.sort_values(by="fp_div", ascending=False).head(10)
```

Note how we specify the attributes that can be used to define subgroups. 
In the above code, we use `boolean_outcomes` because `fp` is boolean. 
The following example, from the example notebook, shows how to use 
`quantitative_outcomes` for a quantitative outcome.

```python
df_census = pd.read_csv('https://raw.githubusercontent.com/divexplorer/divexplorer/main/datasets/census_income.csv')
explorer = DivergenceExplorer(df_census)
value_subgroups = explorer.get_pattern_divergence(
    min_support=0.001, quantitative_outcomes=["PTOTVAL"])
```

### Analyzing subgroups via Shapley values

Returning to our COMPAS example, if we want to analyze what factors 
contribute to the divergence of a particular subgroup, 
we can do so via Shapley values: 

```python
fp_details = DivergencePatternProcessor(FP_fm, 'fp')

pattern = fp_details.patterns['itemset'].iloc[37]
fp_details.shapley_value(pattern)
```

### Pruning redundant subgroups

If you get too many subgroups, you can prune redundant ones via _redundancy pruning_. 
This prunes a pattern $\beta$ if there is a pattern $\alpha$, subset of $\beta$, with a divergence difference below a threshold. 

```python
df_pruned = fp_details.redundancy_pruning(th_redundancy=0.01)
df_pruned.sort_values("fp_div", ascending=False).head(5)
```

##  Algorithms

DivExplorer can multiple frequent-pattern mining algorithms: `fpgrowth`, `apriori`, and `alt_apriori`. 
The default is `fpgrowth`, which is faster; you can switch to `apriori` by specifying `algorithm='apriori'` in the `get_pattern_divergence` method.  The advantage of the `apriori` algorithm is that it can be more memory-efficient. The `alt_apriori` algorithm is memory-efficient but slower.

The `fpgrowth` and `apriori` algorithms are adapted from the [mlxtend](https://rasbt.github.io/mlxtend/) library. 
The `alt_apriori` algorithm is derived from the paper:

> Sergey Brin, Rajeev Motwani, Jeffrey D. Ullman, and Shalom Tsur. “Dynamic Itemset Counting and Implication Rules for Market Basket Data.” In Proceedings of the 1997 ACM SIGMOD International Conference on Management of Data  - SIGMOD ’97, 255–64. Tucson, Arizona, United States: ACM Press, 1997. https://doi.org/10.1145/253260.253325.

We added to the above algorithm options and optimizations that speed up the computation and allow limiting the memory used. 

### `alt_apriori` options

* You can specify the maximum number of items in an itemset, with `max_items` (default: `None`, meaning all items). For instance, if finding itemsets with at most 4 items suffices, you can specify `max_items=4`.

* The dataset is shuffled with a random seed, and for each itemset, only the first `max_instances` instances are considered (default: `None`, meaning all instances).  For example, if you want to use a support limit of 0.01, choosing `max_instances=10000` guarantees that every frequent itemset will be supported by at least 100 instances, allowing for the computation of somewhat reliable divergence values.   If the dataset is very large (in number of instances), this can lead to considerable speedup. 

## Code Contributors

Project Lead: 

- [Eliana Pastor](https://github.com/elianap)

Contributors: 

- [Luca de Alfaro](https://luca.dealfaro.com/)

Other contributors: 

- [Harsh Dadhich]()
