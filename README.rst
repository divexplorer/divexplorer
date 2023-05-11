divexplorer
===========

.. image:: https://img.shields.io/pypi/v/divexplorer.svg
    :target: https://pypi.python.org/pypi/divexplorer
    :alt: Latest PyPI version


DivExplorer
----


Installation
------------
Install using `pip <http://www.pip-installer.org/en/latest/>`__ with:

::

    pip install divexplorer

or, `download a wheel or source archive from
PyPI <https://pypi.org/project/divexplorer/>`__.


Usage
-----
[**Running example - COMPAS dataset**](https://github.com/elianap/divexplorer/blob/main/notebooks/Example_Divergence_analysis_COMPAS.ipynb) - You can find a running example of the usage of the DivExplorer in this notebook.


For the analysis of the divergent classification behavior in subgroups:
```python
from divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer.FP_Divergence import FP_Divergence


# Input: a discretized dataframe with the true class and the predicted class. We specify their column names in the dataframe
# The class_map is a dictionary to specify the positive and the negative class (e.g. {"P":1, "N":0})
fp_diver=FP_DivergenceExplorer(df_discretized, "class", "predicted", class_map=class_map)


#Extract frequent patterns (FP) and compute divergence
##min_support: minimum support threshold
##metrics: metrics=["d_fpr", "d_fnr"] (default metric of interest: False Positive Rate (FPR), False Negative Rate (FNR), Accuracy divergence)
min_sup=0.1
FP_fm=fp_diver.getFrequentPatternDivergence(min_support=min_sup, metrics=["d_fpr", "d_fnr"])
```

The output is a pandas dataframe. Each row indicate a FP with its classification performance and its divergence.

We can then analyze the divergence of FP with respect a metric of interest (e.g. FPR).

```
fp_divergence_fpr=FP_Divergence(FP_fm, "d_fpr")
```

We can sort the itemset for FPR divergence (and visualize the K=10 most divergent ones)
```
K=10
FP_fpr_sorted=fp_divergence_fpr.getDivergence(th_redundancy=0).head(K)
```
or directly get the top K divergent patterns
```
#As a dictionary, where the key are FP and values are their divergence values
topK_dict_fpr=fp_divergence_fpr.getDivergenceTopK(K=10, th_redundancy=0)
#Or as a DataFrame
topK_df_fpr=fp_divergence_fpr.getDivergenceTopKDf(K=10, th_redundancy=0)
```

We can estimate the contribution of each item to the divergence using the notion of Shapley value
```
#Let be itemset_i a FP of interest, for example the one with highest FP_Divergence
itemset_i=list(topK_dict_fpr.keys())[0]
itemset_shap=fp_divergence_fpr.computeShapleyValue(itemset_i)
#Plot shapley values
fp_divergence_fpr.plotShapleyValue(shapley_values=itemset_shap)
#Alternatively, we can just use as input the itemset itself
fp_divergence_fpr.plotShapleyValue(itemset=itemset_i)
```

The itemset can also be inspected using the lattice graph
```
#Plot the lattice graph
#'Th_divergence: if specified, itemsets of the lattice with divergence greater than specified value are highlighted in magenta/squares
##getLower: if True, corrective patterns are highlighted in light blue/diamonds
fig=fp_divergence_fpr.plotLatticeItemset(itemset_i, Th_divergence=0.15, sizeDot="small", getLower=True)
fig.show()
```


DivExplorer allows to identify peculiar behaviors as corrective phenomena.
Corrective items are items that, when added to an itemset, *reduce* the divergence. 
```fp_divergence_fpr.getCorrectiveItems()
```

We can then analyze the influence of each item on the divergence of the entire dataset. 

We can do with:
   - the *individual divergence* of an item. It is simply the divergence of an item in isolation. 
   - the *global divergence* of an item. It a generalization of the Shapley value to the entire set of all items. It captures the role of an item in giving rise to divergence jointly with other attributes. >>> #Compute global shapley value

```
# Individual divergence
individual_divergence_fpr=fp_divergence_fpr.getFItemsetsDivergence()[1]
fp_divergence_fpr.plotShapleyValue(shapley_values=individual_divergence_fpr,sizeFig=(4,5))

# Global divergence 
global_item_divergence_fpr=fp_divergence_fpr.computeGlobalShapleyValue()
fp_divergence_fpr.plotShapleyValue(shapley_values=global_item_divergence_fpr)
```


Note that the evaluation can be perform the analysis of the divergence of a single class of interest:

```
min_sup=0.1
fp_diver_1class=FP_DivergenceExplorer(X_discretized.drop(columns="predicted"),"class", class_map=class_map)
# For example, we analyze the positive rate divergenxe
FP_fm_1class=fp_diver_1class.getFrequentPatternDivergence(min_support=min_sup, metrics=["d_posr", "d_negr"])

```


Authors
-------

[Eliana Pastor](https://github.com/elianap), [Elena Baralis](https://dbdmg.polito.it/wordpress/people/elena-baralis/), [Luca de Alfaro](https://luca.dealfaro.com/)


`divexplorer` was written by `Eliana Pastor <eliana.pastor@polito.it>`_.




