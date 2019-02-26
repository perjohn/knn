# knn

K Nearest Neighbours in a computer vision setting, with implementations for MNIST and CIFAR10.

## Running KNN for MNIST
```
python mnist_knn.py test --k-nn 5
```
Should give an accuracy around 0.97.
Cross validation is run by the command
```
python mnist_knn.py cross-validate
```
Will try a number of k values on a cross-validated training set.

## Running KNN for CIFAR10
```
python cifar10_knn.py test --k-nn 10
```
Should give an accuracy around 0.3.
Cross validation is run by the command
```
python cifar10_knn.py cross-validate
```
Will try a number of k values on a cross-validated training set.

## Running init tests
```
pytest tests
```
