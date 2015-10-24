# Classification

Three classifiers:

1. **Bayesian classifier based on maximum-likelihood estimation**
2. **Bayesian classifer based on Parzen window estimation**
3. **Basic k-nearest neighbor rule**

Tested on three different data sets (corresponding credits and information are in their respective `./bin/*_readme.txt` files):

- **Iris**
- **UCI Wine**
- **Handwritten Digits**

##Usage
In general:
```python3 run.py [-h] [-v] clsf training_filepath test_filepath```

So, if you wanted to use the maximum likelihood classifer on the iris data set, then the command would be `python3 run.py mle ./bin/iris_training.txt ./bin/iris_test.txt`
