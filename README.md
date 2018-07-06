
# BrainStorm

### Description:

BrainStorm is a full connected (feedforward) network library written in C. It supports the following activation functions: sigmoid, relu, tanh and softmax (only at the output layer). It also supports L2 regularization, momentum and three algorithms for adapative learning rate: AdaGrad, RMSProp and Adam.

At this time, parameters for the network must be defined in an input file with the define { } directive. The input file must not contain comments or tabulations. 

The topology for the network can for example be defined with:

```
define {
    topology:[784,30,10]
}
```
The first and last entries of the topology vector are always respectively the number of imputs and outputs of the network.

The mandatory network parameters are the location of the training data, the topology of the network, the split of the training data and the classfication for the training. If no activation function is given, a sigmoid function is assumed for all layers. The default values for the optional parameters are:

```
number of epochs: 30
mini-batch size: 10
learning rate: 0.5
regularization factor: 0
momentum coefficient: 0
```

One can define a deep network with the syntax:
```
define {
    topology:[784,(2~51;30),10]
}
```
This defines a network with 50 hidden layer each with 30 neurons.  Or with different number of neurons for each set of hidden layers like so:
```
define {
    topology:[784,(2~31;30),(32~51;10),10]
}
```

The same can be done to define the activation functions:
```
define {
    activations:[(2~51;sigmoid),softmax]
}
```
This uses a sigmoid function for all hidden layers and a softmax function at the output layer.

To use the same activation function on all layers, one can write:

```
define {
    activations:[sigmoid~]
}
```

The topology, activation function, split, classification, RMSProp and Adam parameters must always be defined as a vector of values with the [ ] syntax. Other parameters are simply scalars.

The values used by the RMSProp method must be defined in the order: [decay rate, delta].
The values used by the Adam method must be defined in the order: [time, step size, decay rate 1, decay rate 2, delta].

If a seperate test data set is provided, then validation data are also created from the training set using the split. Otherwise test data are created from the training set using the split.

The library requires BLAS/LAPACK to compile.
