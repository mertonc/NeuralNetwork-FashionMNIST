# NeuralNetwork-FashionMNIST
A study on neural network that compares different regularization techniques, activation functions, and optimizers to find the optimal model. To summarize, it is important to factor in regularization, activation function, and optimizers to be able to fully unlock neural network’s potential. It is important to note that each dataset is different and requires the data scientist or analyst to make their own research before concluding on the most ‘optimal’ model.

# Notes
The first instinctive observation is the amount of time it takes to run networks with high numbers of epoch. It makes it so that approaching the code takes more care before running. For the results there are multiple findings that basically emphasize the need for trial and error to find the best options for each unique dataset.

The activation function of the neural network effects both the accuracy and execution time. Activation function has the highest impact on the accuracy. Some functions are better fit for some datasets. In the case of the assignment, Sigmoid was not a good option. Execution time varies around a few seconds.

Optimizers are great for increasing accuracy from the baseline neural network. The one that performed the best for the assignment was Adagrad with 85.20&. Other optimizers all performed better than the baseline, making it almost essential for any neural network. The execution time does increase from adding optimizers and is heavily increased for Rprop optimizer.

Regularization also impacts accuracy with less impact on execution time. Each regularization technique that was attempted help boost the accuracy from the baseline. Batch Norm performed the best with an accuracy of 85.20% from the baseline of 71%. The overall execution time stayed around the same for each regularization technique with only a few seconds of difference from 88-92 seconds.
