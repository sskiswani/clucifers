# Classifiers

Three classifiers:

1. **Bayesian classifier based on maximum-likelihood estimation**
2. **Bayesian classifer based on Parzen window estimation**
3. **Basic k-nearest neighbor rule**

Included in this repo are the three different data sets that were used for testing. (corresponding credits and information are in their respective `./bin/*_readme.txt` files):

- **Iris**
- **UCI Wine**
- **Handwritten Digits**

##Usage
Simplified by using the `run.py` script:
```python3 run.py classfier_name path_to_training_data path_to_testing_data [-h] [-v]```

The possible classifier names are:

- `mle` for the MLE Bayesian classifier.
- `parzen` (or `p`) for the Parzen window Bayesian classifer.
- `knn` for the k-nearest neighbors classifier.

So, if you wanted to use the maximum likelihood classifer on the iris data set, then the command would be `python3 run.py mle ./bin/iris_training.txt ./bin/iris_test.txt`.

To customize use of the `classf` module (e.g. make a custom `run.py` script), the module has a run command that can help, and following example demonstrates usage:

```python

def myCustomFileParser(filepath):
    # parse file into a numpy array, with each instance a row in this array.
    # e.g. data[0] corresponds to the first instance's feature vector. 

import classf
classifier = 'parzen'
training_data = './training.txt'
testing_data = './testing.txt'
verbose = True

run(classifier, training_data, testing_data, verbose, myCustomFileParser)
```
