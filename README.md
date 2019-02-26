# knn

K Nearest Neighbours in a computer vision setting, with implementations for MNIST and CIFAR10.

## Running KNN for MNIST
```
python app.py mnist-test --k-nn 5
```
Should give an accuracy around 0.97.
Cross validation is run by the command
```
python app.py mnist-cross-validate
```
Will try a number of k values on a cross-validated training set.

## Running KNN for CIFAR10
```
python app.py cifar10-test --k-nn 10
```
Should give an accuracy around 0.3.
Cross validation is run by the command
```
python app.py cifar10-cross-validate
```
Will try a number of k values on a cross-validated training set.

## Running init tests
```
pytest tests
```
