# Simple Artificial Neural Network

A simple (1 hidden layer) neural network that trains on the included [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Used as an introduction to neural networks/machine learning and [armadillo](https://arma.sourceforge.net/).

Currently a simple cli application that processes the MNIST files, trains on
the training files and lastly tests on the test files.

Currently gets to ≈90% accuracy on the training data after 200 iterations after which the gains are minimal — ≈94% at
1000-2000 iterations. Currently gets an accuracy of 93,79% at 1000 iterations on
the test data.

Builds on Windows 11 using clang and vs.

<details>
  <summary>
    <strong>TODO</strong>
  </summary>

- Visualization of the MNIST images and their respective labels.
- Implementation of 2nd hidden layer and other improvements.

</details>


## Licenses
Main network licensed under the MIT License.

[Armadillo](https://arma.sourceforge.net/) and [OpenBLAS](https://www.openblas.net/) are licensed under Apache-2.0 and BSD-3-Clause,
respectively.

DatasetProcessing is a slightly modified version of Baptiste Wicht's [mnist](https://github.com/wichtounet/mnist). MIT License.

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is not my property.

