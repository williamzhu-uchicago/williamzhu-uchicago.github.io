---
layout: post
title: Andreas Muller & Sarah Guido - Introduction to Machine Learning with Python - Chapter 2.3
---

# Naive Bayes Classifiers

**Naive Bayes classifiers** are another family of classifiers that are fast in training but have weaker generalization performance. The calssifiers learn parameters by looking at each features individually and collect simple er-class statistics from it. Sklearn provides `GaussianNB`, `BernoulliNB`, and `MultinomialNB`. Their names suggest the type of data that they are applied to: BernoulliNB and MultinomialNB are used on count data with the latter usually offering a better performance, and GaussianNB is used on high-dimensional or continuous data.

**BernoulliNB** classifier is the simplest. It counts how often every feature of each class is not zero. Consider the following example. We have 4 data points (rows) and 4 features (columns). For class 0, the first feature appears non-zero 0 time, the second feature appears non-zero 1 time, and so on.


```python
import numpy as np
X = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])

for label in np.unique(y):
    print(f"Feature counts of class {label}: {X[y == label].sum(axis=0)}")
```

    Feature counts of class 0: [0 1 0 2]
    Feature counts of class 1: [2 0 2 1]
    

**MultinomialNB** is different in a way that it stores the mean of each feature for each class, and **GaussianNB** stores both the mean and variance of each feature.

To make a predication, the test data is compared to the statistics for each of the classes, and the best-matching class is predicted as the data's class.

BernoulliNB and MultinomialNB take a single parameter, `alpha`. The algorithm adds to the data $\alpha$-many virtual data points that have positive values for all features to smoothen the statistics. The higher the alpha, the more smooth is the statsitics and hence the less complex is the model.
