[![PyPI](https://img.shields.io/pypi/v/divexplorer)](https://pypi.org/project/divexplorer/)
[![Downloads](https://pepy.tech/badge/divexplorer)](https://pepy.tech/project/divexplorer)

# DivExplorer

- [DivExplorer](#divexplorer)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Notebooks](#notebooks)
    - [Package](#package)
  - [Paper](#paper)
  - [Contributing](#contributing)

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
You can also [run the notebook directly on Colab](https://colab.research.google.com/drive/1lDqqssBusiFHjgR6EciuWNT55hO4_X0o?usp=sharing). 

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
subgroups = fp_diver.get_pattern_divergence(min_support=0.001, quantitative_outcomes=["PTOTVAL"])
subgroups.sort_values(by="PTOTVAL_div", ascending=False).head(10)
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

## Papers

The original paper is:

> [Looking for Trouble: Analyzing Classifier Behavior via Pattern Divergence](https://divexplorer.github.io/static/DivExplorer.pdf). [Eliana Pastor](https://github.com/elianap), [Luca de Alfaro](https://luca.dealfaro.com/), [Elena Baralis](https://dbdmg.polito.it/wordpress/people/elena-baralis/). In Proceedings of the 2021 ACM SIGMOD Conference, 2021.

You can find more papers in the [project page](https://divexplorer.github.io/).

## Code Contributors

Project lead:

- [Eliana Pastor](https://github.com/elianap)

Other contributors: 

- [Luca de Alfaro](https://luca.dealfaro.com/)
- [Harsh Dadhich]()

# Documentation

<a id="divexplorer"></a>

# divexplorer

<a id="divexplorer.DivergenceExplorer"></a>

## DivergenceExplorer Objects

```python
class DivergenceExplorer()
```

<a id="divexplorer.DivergenceExplorer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(df, is_one_hot_encoding=False)
```

**Arguments**:

- `df`: pandas dataframe.  The columns that one wishes to analyze with divexplorer should have discrete values.
- `is_one_hot_encoding`: boolean. If True, the dataframe attributes that one wishes to analyze are already one-hot encoded.

<a id="divexplorer.DivergenceExplorer.get_pattern_divergence"></a>

#### get\_pattern\_divergence

```python
def get_pattern_divergence(min_support: float,
                           boolean_outcomes: list = None,
                           quantitative_outcomes: list = None,
                           attributes: list = None,
                           FPM_algorithm="fpgrowth",
                           show_coincise=True)
```

Computes the divergence of the specified outcomes.  One can specify two types of outcomes: boolean and quantitative.

The difference lies mainly in the way in which the statistical significance is computed: in both cases, we use
the Welch's t-test, but for boolean outcomes, we consider the outcomes as Bernoulli random variables. 
One can specify multiple outcomes simultaneously, as a way to speed up the computation when multiple divergences are needed 
(compared to computing them one by one).

**Arguments**:

- `min_support`: minimum support value for the pattern
- `boolean_outcomes`: list of boolean outcomes
- `quantitative_outcomes`: list of quantitative outcomes
- `attributes`: list of attributes to consider
- `FPM_algorithm`: algorithm to use for frequent pattern mining
- `show_coincise`: if True, the output is more concise, returning only the average, the divergence and the t value

<a id="outcomes"></a>

# outcomes

<a id="outcomes.get_false_positive_rate_outcome"></a>

#### get\_false\_positive\_rate\_outcome

```python
def get_false_positive_rate_outcome(y_trues, y_preds, negative_value=0)
```

Returns boolean outcome for the false positive rate. 1 if it is a false positive,

0 if it is a true negative, np.nan otherwhise.

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values.
- `negative_value`: value of the negative class.

**Returns**:

boolean outcome column values for the false positive rate.

<a id="outcomes.get_false_negative_rate_outcome"></a>

#### get\_false\_negative\_rate\_outcome

```python
def get_false_negative_rate_outcome(y_trues, y_preds, positive_value=1)
```

Returns boolean outcome for the false negative rate.

1 if it is a false negative, 0 if it is a true positive, np.nan otherwhise.

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values.
- `positive_value`: value of the positive class.

**Returns**:

boolean outcome column values for the false negative rate.

<a id="outcomes.get_accuracy_outcome"></a>

#### get\_accuracy\_outcome

```python
def get_accuracy_outcome(y_trues, y_preds, negative_value=0, positive_value=1)
```

Returns boolean outcome for the accuracy rate. 1 if it is correct, 0 if it is incorrect.

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values.
- `negative_value`: value of the negative class.
- `positive_value`: value of the positive class.

**Returns**:

boolean outcome column values for the accuracy rate.

<a id="outcomes.get_true_positives"></a>

#### get\_true\_positives

```python
def get_true_positives(y_trues, y_preds, positive_value=1)
```

Returns true positives. True if it is a true positive, false otherwise.

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values
- `positive_value`: value of the positive class.

**Returns**:

boolean outcome column values for the true positive rate.

<a id="outcomes.get_true_negatives"></a>

#### get\_true\_negatives

```python
def get_true_negatives(y_trues, y_preds, negative_value=1)
```

Returns true negatives. True if it is a true negative, false otherwise.

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values.
- `negative_value`: value of the negative class.

**Returns**:

boolean outcome column values for the true negative rate.

<a id="outcomes.get_false_positives"></a>

#### get\_false\_positives

```python
def get_false_positives(y_trues, y_preds, negative_value=1)
```

Returns false positives. True if it is a false positive, false otherwise

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values.
- `negative_value`: value of the negative class.

**Returns**:

boolean outcome column values for the false positive rate.

<a id="outcomes.get_false_negatives"></a>

#### get\_false\_negatives

```python
def get_false_negatives(y_trues, y_preds, positive_value=1)
```

Returns false negatives. True if it is a false negative, false otherwise.

**Arguments**:

- `y_trues`: true values (e.g., df['y_true'])
- `y_preds`: predicated values.
- `positive_value`: value of the positive class.

**Returns**:

boolean outcome column values for the false negative rate.

<a id="pattern_processor"></a>

# pattern\_processor

<a id="pattern_processor.DivergencePatternProcessor"></a>

## DivergencePatternProcessor Objects

```python
class DivergencePatternProcessor()
```

Class to process patterns and compute Shapley values.

<a id="pattern_processor.DivergencePatternProcessor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(patterns, metric)
```

**Arguments**:

- `patterns`: dataframe patterns
- `metric`: name of the field (without final `_div`) used for divergence

<a id="pattern_processor.DivergencePatternProcessor.dict_len_pattern_divergence_representation"></a>

#### dict\_len\_pattern\_divergence\_representation

```python
def dict_len_pattern_divergence_representation()
```

Define an intermediate representation of the patterns dataframe in the form

Len itemset -> {pattern: divergence}

**Returns**:

Mapping from item length to item divergence.

<a id="pattern_processor.DivergencePatternProcessor.shapley_value"></a>

#### shapley\_value

```python
def shapley_value(pattern=None, row_idx=None)
```

Compute the Shapley value of a pattern
We can specify the pattern either directly by specifying the pattern (frozen set) or row_idx of the pattern in the patterns dataframe

**Arguments**:

- `pattern` _frozen set_ - list of items - if None, row_idx must be provided
- `row_idx` _int_ - row index of the pattern in the patterns dataframe - if None, pattern must be provided

**Returns**:

  (dict) Shapley value of the pattern - {item: shapley value} for each item in the pattern

<a id="pattern_processor.DivergencePatternProcessor.plot_shapley_value"></a>

#### plot\_shapley\_value

```python
def plot_shapley_value(pattern: frozenset = None,
                       row_idx: int = None,
                       shapley_values: dict = None,
                       figsize: tuple = (4, 3),
                       abbreviations: dict = {},
                       sort_by_value: bool = True,
                       height: float = 0.5,
                       linewidth: float = 0.8,
                       labelsize: int = 10,
                       title: str = "",
                       x_label="",
                       name_fig: str = None,
                       save_fig: bool = False,
                       show_figure: bool = True)
```

Plot the Shapley value of a pattern.

Specify either pattern or row_idx or shapley_value.

**Arguments**:

- `pattern`: list of items
- `row_idx`: row index of the pattern in the patterns dataframe
- `shapley_values`: dictionary of pattern scores: {pattern: score}
- `figsize`: figure size
- `abbreviations`: dictionary of abbreviations to replace in the patterns - for visualization purposes
- `sort_by_value`: sort the Shapley values by value
- `height`: height of the bars
- `linewidth`: width of the bar border
- `labelsize`: size of the labels
- `title`: title of the plot
- `x_label`: x label
- `name_fig`: name of the figure
- `save_fig`: save the figure
- `show_figure`: show the figure

<a id="pattern_processor.DivergencePatternProcessor.redundancy_pruning"></a>

#### redundancy\_pruning

```python
def redundancy_pruning(th_redundancy)
```

Prune the patterns that are redundant with respect to the divergence

**Arguments**:

- `th_redundancy`: threshold for redundancy

**Returns**:

a Pandas dataframe containing patterns without redundancy
Let I and  I - {item i} be two patterns   (for example,  {sex=Male, age=<25} and {sex=Male})
If exist an item i such that it absolute marginal contribution is lower than a threshold epsilon,
i.e. abs( divergence(I) - divergence(I - {item i}) <= epsilon
We can prune I. The pattern 𝐼 - {item i} captures the divergence of pattern 𝐼, since the inclusion of the item i only slightly alters the divergence
In the example, we would keep just sex=Male
We proceed in this way for all the patterns.

<a id="pattern_processor.DivergencePatternProcessor.get_patterns"></a>

#### get\_patterns

```python
def get_patterns(th_redundancy=None, sort_by_divergence=True)
```

Return the patterns

**Arguments**:

- `th_redundancy`: threshold for redundancy - if None, no redundancy pruning
- `sort_by_divergence`: sort the patterns by divergence

**Returns**:

a Pandas dataframe containing the patterns and their divergence

<a id="pattern_processor.DivergencePatternProcessor.global_shapley_value"></a>

#### global\_shapley\_value

```python
def global_shapley_value()
```

Compute the Global Shapley value of the patterns

The Global Shapley value is a generalization of the Shapley value to the entire set of all items.
It captures the role of an item in giving rise to divergence jointly with other attributes.

**Returns**:

A dictionary associating each item to its Global Shapley value.

<a id="shapley_value"></a>

# shapley\_value

<a id="shapley_value.compute_shapley_value"></a>

#### compute\_shapley\_value

```python
def compute_shapley_value(pattern: frozenset, item_score: dict)
```

Compute the Shapley value of a subset of items

**Arguments**:

- `pattern`: list of items
- `item_score`: dictionary of pattern scores: len(pattern) -> {pattern: score}

<a id="shapley_value.attribute"></a>

#### attribute

```python
def attribute(item)
```

Returns the attribute of an item.

**Arguments**:

- `item` _str_ - item in the form of 'attribute=value'

**Returns**:

- `str` - attribute of the item

<a id="shapley_value.weight_factor_global_shapley_value"></a>

#### weight\_factor\_global\_shapley\_value

```python
def weight_factor_global_shapley_value(lB, lA, lI, prod_mb)
```

Returns the weight factor for the global shapley value.

<a id="shapley_value.global_itemset_divergence"></a>

#### global\_itemset\_divergence

```python
def global_itemset_divergence(I, scores_l, attributes, cardinality_attributes)
```

Returns the global divergence of an itemset.

**Arguments**:

- `I` _set_ - itemset
- `scores_l` _dict_ - dictionary of len itemset, itemsets and their divergence
- `attributes` _list_ - list of attributes (from frequent items)
- `card_map` _dict_ - dictionary of attributes and their cardinality, i.e., how many values they can take (from frequent items)

