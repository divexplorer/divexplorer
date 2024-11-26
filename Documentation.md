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

