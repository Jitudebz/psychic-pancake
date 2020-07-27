# psychic-pancake
This repository will contain all the analytics projects reports, data, code, ppts and other files from analytics tools which I have worked on during my post graduation.

TensorFlow is a free and open-source software library for dataflow and differentiable programming across a range of tasks. It is a symbolic math library and is also used for machine learning applications such as neural networks. 
It is was developed by the Google Brain team for internal Google use. It was released under the Apache License 2.0 on November 9, 2015. Currently it is used for both research and production at Google.

In this project, we will see how to classify the MNIST Handwritten digits using TensorFlow and Keras libraries with the help of Neural Network algorithm.
The MNIST Handwritten Digits is a dataset for evaluating machine learning and deep learning models on the handwritten digits classification problem, it is a dataset of 60,000 small square 28x28 pixel grayscale images of handwritten single digits between 0 and 9

http://yann.lecun.com/exdb/mnist/

Neural Networks Overview

Neural networks are used as a method of deep learning, one of the manysubfields of artificial intelligence.
In an attempt to simulate the way the human brain works, individual ‘neurons’ are connected in layers, with weights assigned to determine how the neuron responds when signals are propagated through the network.
Due to advancements in hardware development and major breakthroughs we have built advanced machines to match and exceed the capabilities of humans at performing certain tasks.
One such task is object recognition.

Theory Behind Neural Networks

Though machines have historically been unable to match human vision, recent advances in deep learning have made it possible to build neural networks which can recognize objects, faces, text, and even emotions.
Neural networks consist of a number of artificial neurons which each process multiple incoming signals and return a single output signal. The output signal can then be used as an input signal for other neurons.
The weights are the neuron’s internal parameters. Both input vector and weights vector contain the same number of values, so we can use them to calculate a weighted sum.
Now as long as the result of the weighted sum is a positive value, the neuron’s output is the weighted sum value. But if the weighted sum is a negative value, we ignore that negative value and the neuron generates an output of 0 instead. This operation is called a Rectified Linear Unit (ReLU).


Now we are going to build the model or in other words the neural network that will train and learn how to classify these images.
Its worth noting that the layers are the most important thing in building an artificial neural network since it will extract the features of the data.
First and foremost, we start by creating a model object that lets you add the different layers.
Second, we are going to flatten the data which is a image pixels in this case. So the images are 28 x 28 dimensional we need to make it 1 x 784 dimensional so the input layer of the neural network can read it and deal with it.

Third, we define input and a hidden layer with 128 neurons and an activation function which is ReLU function.
And the last thing we create the output layer with 10 neurons and a sigmoid/softmax activation function that will transform the score returned by the model to a value so it will be interpreted by humans

Since we finished building the neural network we need to compile the model by adding some few parameters that will tell the network how to start the training process.
First, we add the optimizer (here we used Adam Optimizer), which will update the parameter of the neural network to fit our data.
Second, the loss function that will tell you the performance of your model.
Third, the metrics which give indicative tests of the quality of the model.
We train our model using the fit sub package and feed it with the training data and the labeled data that correspond to the training dataset and how many epoch should run or how many times should make a guess.

The test accuracy has reached 97.27% which is pretty good.


In this project, using neural networks from TensorFlow library for handwritten recognition we could determine around 10000 new handwritten characters (numbers) which is really shows the potential of ANN in the field of classification and recognition.
Results shows the highest classification accuracy and lowest classification time in comparison with other machine learning algorithms.

https://github.com/tensorflow/
The MNIST Dataset consists of 70000 images http://yann.lecun.com/exdb/mnist/





