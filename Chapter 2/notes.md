# Classification

## MNIST

MNIST is a dataset which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau.


In this chapter, we will train a machine learning model that can classify if a given image is of a 5 or not-5.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
```

Let's create two variables: X and y for features and labels respectively:

```python
X, y = mnist.data, mnist.target
```

There are 70,000 images in total, and each image has 784 features / pixels. This means that each image has a resolution of 28x28.

Let's split this into training set and test set:

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```
## Training a Binary Classifier

Let's start training a binary classifier. A binary classifier is basically a type of classifier where our model has to choose from just two options, like cat or dog, five or not-5, comedy or horror etc. In our case, we need to train a binary classifier that classifies whether a given image is of a 5 or not-5. Let's narrow down our labels into just 5 or not 5:

```python
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')
```

This means that y_train_5 will have the value of True if a given image has the label of '5', and False if it's not 5.

Now let's train a SGD Classifier. An SGD Classifier uses gradient descent to tweak the parameters of the model, but instead of tweaking all the parameters at once, it looks at one instance / gradient at a time, so it is very efficient for larger datasets. I will cover more about gradient descent in the next chapter.

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)
```

Finally! You’ve built the model. Everything perfect? Well, not exactly. Now we need to measure its performance, and it might sound easy in words, but measuring the performance of a classifier is significantly harder than a regression model. There are a lot of things you have to measure, we will cover these right now:

* Confusion Matrix
* Precision Score
* Recall Score
* F1 Score
* ROC (Receiver Operating Characteristic)
* ROC-AUC (Area Under the Curve)

Okay! Let's get right into this.

### Confusion Matrix

To first understand how a confusion matrix works, or in general, how to measure the performance of a classifier, you must first understand these terms:

**Note**: I will define them in our context (5 or not-5)

* **True Positive**: The model said that an image is of a 5 and it was a 5. The model was correct

* **True Negative**: The model said that an image is not a 5 and it was not a 5. The model was correct.

* **False Positive**: The model said that the image is a 5 but it was not 5. The model was incorrect.

* **False Negative**: The model said that the image is not a 5 but it was a 5. The model was incorrect.

Here is what a **Confusion Matrix** looks like:

[[58391, 687],

[1891,   3530]]

The top left number is the number of **True Negatives**.

The top right number is the number of **False Positives**.

The bottom left number is the number of **False Negatives**.

The bottom right number is the number of **True Positives**.

To first see the confusion matrix of the model, we should first do ```cross_val_predict``` to first make some predictions:

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

### Precision

Precision tells you: "Out of all the predicted positives, how many were actually positives and correct?"

Precision = $\frac{TP}{TP+FP}$.

### Recall

Recall tells you: "Out of all the actual positives, how many did the model catch?"

Recall = $\frac{TP}{TP+FN}$.

### Precision and Recall

To measure the precision and recall of a model, you can use the `precision_score` and `recall_score` methods provided by sklearn.metrics respectively:

```python
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_train_pred))

print(recall_score(y_train_5, y_train_pred))
```
### F1 Score

The F1 score is a harmonic mean of precision and recall meaning the classifier will only get a high F1 score if both precision and recall are high.

To measure F1 score you can use the `f1_score` function provided by `sklearn.metrics` library:

```python
from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))
```
The F1 score favors classifiers that have similar precision and recall.

### Precision/Recall Trade-off

To understand this tradeoff, let's first understand how the SGDClassifier makes it classification decisions. For each instance, it computes a score based on a decision function. If that score is greater than a threshold, it assigns the instance to the positive class, otherwise it assigns it to the negative class.

This results in a problem: You have to balance precision and recall, but it's very hard to do so. If you have 99% precision, recall could be very low!

### ROC Curve

The ROC Curve plots the Recall against FPR (False Positive Rate). But there is a tradeoff, the higher the recall, the more false positives the classifier produces.

Another way to compare classifiers is to measure the area under the curve (AUC). A perfect classifier will have a ROC-AUC of 1. Scikit-learn provides the `roc_auc_score` function in the `sklearn.metrics` library to measure the ROC-AUC of a model.

# Conclusion

Finally! You have done it! You learnt about SGDClassifier, we’ve covered too many performance metrics to count and now you should be pretty comfortable making some pretty good classification projects! But this is not the end! In the next chapter, we will cover the theory behind these machine learning models, which we’ve treated as black boxes for the last two chapters. I’ll explain how they make decisions and what happens under the hood, which is really useful if you want to actually deeply understand machine learning, or if you want to go into research (Like ME!). Overall, this was a lot of fun, see you soon!
