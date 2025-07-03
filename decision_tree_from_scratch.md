
# Building a Decision Tree from Scratch (with Math)

This guide walks you through creating a decision tree classifier from scratch, including the mathematical concepts behind it.

---

## 🌳 1. What Is a Decision Tree?

A decision tree is a supervised learning model used for classification and regression. It splits data into subsets based on feature values, creating a tree-like structure of decisions.

---

## 📊 2. Core Concept: Information Gain

To choose which feature to split on, decision trees use a metric like **Information Gain**, which is based on **Entropy**.

### Entropy

Entropy measures the impurity or disorder of a dataset. It’s defined as:

\[
H(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
\]

Where:
- \( S \): dataset
- \( c \): number of classes
- \( p_i \): proportion of class \( i \)

### Information Gain

Information Gain tells us how much entropy decreases after a split:

\[
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
\]

Where:
- \( A \): the attribute to split on
- \( S_v \): subset of data where attribute \( A = v \)

---

## 🔨 3. Algorithm Steps

### Step 1: Calculate Entropy of the Dataset

```python
import numpy as np

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))
```

### Step 2: Split Dataset on Each Feature and Calculate Info Gain

```python
def info_gain(X_column, y):
    # Possible values in the column
    values = np.unique(X_column)
    total_entropy = entropy(y)

    weighted_entropy = 0
    for val in values:
        subset_y = y[X_column == val]
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)

    return total_entropy - weighted_entropy
```

### Step 3: Choose the Best Feature to Split

```python
def best_split(X, y):
    best_gain = -1
    best_feature = None

    for i in range(X.shape[1]):
        gain = info_gain(X[:, i], y)
        if gain > best_gain:
            best_gain = gain
            best_feature = i

    return best_feature
```

### Step 4: Recursively Build the Tree

```python
class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, *, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label

def build_tree(X, y):
    # If pure or empty, return leaf node
    if len(np.unique(y)) == 1:
        return Node(label=y[0])

    if X.shape[1] == 0:
        # No features left to split
        unique, counts = np.unique(y, return_counts=True)
        return Node(label=unique[np.argmax(counts)])

    feature = best_split(X, y)
    values = np.unique(X[:, feature])

    if len(values) == 1:
        return Node(label=y[0])

    nodes = {}
    for val in values:
        idx = X[:, feature] == val
        subtree = build_tree(X[idx], y[idx])
        nodes[val] = subtree

    return Node(feature=feature, value=nodes)
```

---

## 🧪 4. Using the Tree to Predict

```python
def predict(tree, x):
    while tree.label is None:
        val = x[tree.feature]
        if val in tree.value:
            tree = tree.value[val]
        else:
            return None  # unknown value
    return tree.label
```

---

## ✅ Summary

- Entropy quantifies impurity
- Information Gain chooses the best split
- Build the tree recursively
- Make predictions by traversing the tree

This approach helps you understand how decision trees work internally and builds strong intuition for more advanced tree-based models like Random Forests and Gradient Boosted Trees.

---

*You can extend this to handle numerical features, pruning, or create ensembles like Random Forest and AdaBoost!*
