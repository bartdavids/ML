# ML

I'm practicing machine learning application in Python and store some projects here.

The data map stores practice data (MNIST only for now)

# Projects (.py files)

NN
- .py file where a simple neural network can be build, trained and run
- for those that like to read top-down
- ReLu blows out in matmul (@) when using GD (not when using adam)

NN_as_class
- .py file containing the neural network builder and trainer as a class
- Exaple use in NN_as_class_input

NN_as_class_input
- Example use of the NN_as_class file

# Known issues
- ReLu/linear activation functions causes matmul to create inf's (and consequently NaNs) when using gradient descent (not when using ADAM)

# Things I like to add
- Batch training
- Better tracking of performace
