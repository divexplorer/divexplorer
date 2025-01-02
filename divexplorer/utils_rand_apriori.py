import pandas as pd
import numpy as np


def get_items(df, columns=None, support_threshold=0.01):
    """Returns a dictionary, mapping every column name to the list of values that have
    a support greater than the threshold in the column. The returned elements are in 
    order from greatest to least support."""
    cs = columns or df.columns
    d = {}
    for c in cs:
        vc = df[c].value_counts(normalize=True)
        d[c] = vc[vc > support_threshold].index.tolist()
    return d


def get_frequent_items(df, columns=None, support_threshold=0.01):
    """Returns a dictionary, mapping every column name to the list of values that have
    a support greater than the threshold in the column. The returned elements are in 
    order from greatest to least support."""
    cs = columns or df.columns
    d = {}
    for c in cs:
        vc = df[c].value_counts(normalize=True)
        vc = vc[vc > support_threshold]
        d[c] = [(v, vc[v]) for v in vc.index]
    return d


class Node(object):
    def __init__(self, num_counts=1, start_time=0, stop_time=1, is_small=True, 
                 subtree_size=1):
        """
        @param num_counts: the number of counts to keep.
        @param start_time: time in which the node was added to the tree. 
        @param stop_time: time in which the node stops counting. If not null, it is still counting.
        @param is_small: the node is small.
        @param subtree_size: the size of the subtree rooted at the node that is counting, including the node.  
        A node is counted until it has a set stop time, after which no increments occur. 
        """
        self.start_time = start_time
        self.stop_time = stop_time
        self.counting = True
        self.subtree_size = subtree_size
        self.is_small = is_small
        self.count = 0
        self.count_not_nan = np.zeros(num_counts)
        self.totals = np.zeros(num_counts)
        self.totals_squared = np.zeros(num_counts)
        
    def inc(self, xs, current_time):
        """Increments the node, returning whether it stopped counting.
        @param x: the value to increment the node by.
        @param current_time: the current time.
        """
        if current_time > self.stop_time:
            self.counting = False
            return True # It stopped. 
        if self.counting:
            self.count += 1
            self.count_not_nan += (1 - np.isnan(xs).astype(int))
            self.totals += np.nan_to_num(xs, nan=0)
            self.totals_squared += np.nan_to_num(xs ** 2, nan=0)
            return False 
            
    def is_counting(self, current_time):
        """Checks whether the node is counting.  Returns 
        a tuple (is_counting, has_stopped), where: 
        - is_counting is a boolean indicating whether the node is counting.
        - has_stopped is a boolean indicating whether the node has just stopped counting,
          so that the caller can decrease the subtree size of the parent nodes.
        """
        if self.counting and current_time > self.stop_time:
            self.counting = False
            return (False, True)
        return (self.counting, False)

    @property
    def support(self):
        return self.count / (self.stop_time - self.start_time)    
        
    def __repr__(self):
        return f'(is_small={self.is_small}, T0={self.start_time}, Tstop={self.stop_time}, c={self.counts}, t={self.totals})' 
    
    
class Explorer(object):
    
    def __init__(self, 
                 df, 
                 support_threshold=0.01, 
                 chunk_size=1000, 
                 columns=None, 
                 target_columns=None,
                 items_ascending=True, 
                 depth_limit=None, 
                 itemset_limit=None,
                 count_limit=None, 
                 promotion_check=10, 
                 promotion_tolerance=0.9,
                 ):
        """Initializes the Explorer object.
        @param df: the dataframe to analyze.
        @param support_threshold: the minimum support for an item to be considered frequent.
        @param chunk_size: the number of rows to process at once.
        @param columns: the columns to consider in the analysis.
        @param target_list: list of target columns.
        @param items_ascending: whether items are sorted in ascending order of frequency during analysis. 
        @param depth_limit: the maximum depth of the itemsets.
        @param itemset_limit: the maximum number of itemsets to consider.
        @param count_limit: the maximum amount of count to do for each item. 
        @param promotion_check: the number of counts to check for promotion.
        @param promotion_tolerance: the tolerance for promotion.
        """
        assert target_columns is not None
        self.df = df
        self.count_limit = count_limit or len(df)
        self.target_columns = target_columns
        self.num_counts = len(target_columns)
        self.columns = columns or [c for c in df.columns if c not in target_columns]
        self.chunk_size = chunk_size
        self.support_threshold = support_threshold
        self.items_ascending = items_ascending
        self.promotion_check = promotion_check
        self.promotion_tolerance = promotion_tolerance
        self.depth_limit = depth_limit
        self.itemset_limit = itemset_limit
        self.num_large_nodes = 0
        self.frequent_items = get_frequent_items(
            df, 
            columns=self.columns,
            support_threshold=support_threshold)
        self.tree = {}
        self.item_frequencies = {(c, v): f for c, vfs in self.frequent_items.items() for v, f in vfs}
        self.items = list(self.item_frequencies.keys())
        self.items.sort(key=lambda x: self.item_frequencies[x], reverse=not self.items_ascending)
        self.num_items = len(self.items)
        self._init_tree()        
        
    def repr_item(self, i):
        return f'{self.items[i][0]}={self.items[i][1]}'
    
    def _repr_itemset(self, itemset):
        return "(" + ", ".join([self.repr_item(i) for i in itemset]) + ")"
        
    def __repr__(self):
        return "\n".join([f'{self._repr_itemset(itemset)} : {node}' for itemset, node in self.tree.items()])
    
        
    def _init_tree(self):
        self.tree[()] = Node(num_counts=self.num_counts, is_small=False, stop_time=self.count_limit,
                             subtree_size=self.num_items + 1) 
        for i in range(self.num_items):
            self.tree[(i,)] = Node(num_counts=self.num_counts, is_small=False, stop_time=self.count_limit)
        self.num_large_nodes = self.num_items + 1
        # Adds nodes to the item nodes, since we may need to count them. 
        for i in range(self.num_items):
            self._add_children((i,), 0)
                                
                                
    def _get_row_item_indices(self, row):
        """Returns the indices of the items in a row, sorted according to the support."""
        row_items = [(c, row[c]) for c in row.index if c in self.columns and (c, row[c]) in self.item_frequencies]
        row_indices = [self.items.index(item) for item in row_items]
        row_indices.sort()
        return row_indices
    
        
    def _increment_tree(self, item_list, row_items, xs, current_time):
        """Increments the counts in the tree.
        @param item_list: list of items defining the node to increment. 
        @param row_items: list of items defining the remaining row items to process. 
        @param x: the value of the target variable.
        Here, node is the root note to increment, and items is the list of items. 
        As nodes are represented as list of items (their path from the root), the node is also a list of items."""
        item_list_tuple = tuple(item_list)
        node = self.tree[item_list_tuple]
        if node.subtree_size == 0:
            return
        if node.inc(xs, current_time):
            # The subtree size decreased. 
            node.subtree_size -= 1
            for k in range(len(item_list_tuple)):
                self.tree[item_list_tuple[:k]].subtree_size -= 1
        if node.subtree_size > 0:
            for i, item in enumerate(row_items):
                if tuple(item_list + [item]) in self.tree:
                    self._increment_tree(item_list + [item], row_items[i + 1:], xs, current_time)

    
    def _process_row(self, row, current_time):
        """Reads and processes a row, incrementing the counts."""
        row_indices = self._get_row_item_indices(row)
        self._increment_tree([], row_indices, np.array(row[self.target_columns], dtype=float), current_time)
        
        
    def _maybe_large(self, node, current_time):
        """Returns whether the node is a candidate to be large."""
        # For the moment, we use a sure rule. 
        # Later we can: 
        # - use a statistical test.
        # This is the true test.
        if node.count >= self.support_threshold * self.count_limit:
            return True
        # This is the heuristic test.
        delta_time = current_time - node.start_time
        return node.count >= 10 and node.count >= (delta_time * self.support_threshold * self.promotion_tolerance)
            

    def _can_add_children(self, itemset, current_time):
        """Returns whether we can add children to the node."""
        return (
            (self.depth_limit is None or len(itemset) < self.depth_limit)
            and 
            (self.itemset_limit is None or self.num_large_nodes < self.itemset_limit))


    def _add_children(self, itemset, current_time):
        """Adds children to a node that has been promoted to large.
        Returns whether a node has been added."""
        # We need to generate its list of children. 
        # Keeps track of the columns in the itemset. 
        columns = [self.items[i][0] for i in itemset]
        added = False
        for j in range(self.num_items):
            column = self.items[j][0]
            if j not in itemset and column not in columns:
                candidate_itemset = list(itemset) + [j]
                candidate_itemset.sort()
                candidate_itemset = tuple(candidate_itemset)
                if candidate_itemset not in self.tree:
                    # We need to check if all parents of candidate_itemset are large.
                    parents_are_large = True
                    for k in range(len(candidate_itemset)):
                        parent_itemset = candidate_itemset[:k] + candidate_itemset[k + 1:]
                        if parent_itemset not in self.tree or self.tree[parent_itemset].is_small:
                            parents_are_large = False
                            break
                    if parents_are_large:
                        self.tree[candidate_itemset] = Node(
                            num_counts=self.num_counts, 
                            start_time=current_time, 
                            stop_time=current_time + self.count_limit)
                        added = True
                        # Increments the subtree size of the parents. 
                        for k in range(len(candidate_itemset)):
                            self.tree[candidate_itemset[:k]].subtree_size += 1
        return added


    def _update_tree(self, current_time):
        """Updates the tree, promoting and adding nodes. 
        Returns whether we can stop the analysis."""
        can_stop = True
        for itemset in list(self.tree.keys()):
            node = self.tree[itemset]
            if node.is_small and self._maybe_large(node, current_time):
                node.is_small = False
                self.num_large_nodes += 1
                if self._can_add_children(itemset, current_time):
                    if self._add_children(itemset, current_time):
                        can_stop = False # Can't stop if we add a node.
            is_counting, has_stopped = node.is_counting(current_time)
            if has_stopped:
                node.subtree_size -= 1
                for k in range(len(itemset)):
                    self.tree[itemset[:k]].subtree_size -= 1
            if is_counting:
                can_stop = False # Can't stop if we have a node that is not stopped.
        return can_stop
    
    
    def compute_frequent_itemsets(self, seed=1):
        """Computes the frequent itemsets.
        @param seed: seed for dataset shuffling."""
        df_shuffled = self.df.sample(frac=1, random_state=seed)
        epochs = 0
        n = 0
        while True:
            for i, row in df_shuffled.iterrows():
                n += 1
                self._process_row(row, n)
                if n % self.chunk_size == 0:
                    if self._update_tree(n):
                        return
            epochs += 1
            if self._update_tree(epochs * len(df_shuffled)):
                return

            
    @property
    def frequent_itemsets(self):
        """Generates the frequent itemsets, returning the itemset, and tree node, for each."""
        for itemset, node in self.tree.items():
            if node.count < self.support_threshold * self.count_limit:
                continue
            yield itemset, node
            
            
    @property
    def root(self):
        return self.tree[()]
      