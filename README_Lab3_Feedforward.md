# Lab 3: Implementing a Feedforward Neural Network using TensorFlow/PyTorch

## Aim
To implement a feedforward (multilayer) neural network using PyTorch and TensorFlow/Keras, and train it to learn the logical **AND gate**.

## Main Topic

A **Feedforward Neural Network (FNN)**, also called a Multilayer Perceptron (MLP), is an extension of the simple Perceptron. Unlike a single Perceptron, which can only solve linearly separable problems, an FNN has one or more **hidden layers** between the input and output layers. Information flows only in one direction — forward, from input to output — which is why it's called "feedforward."

Each neuron in a layer is connected to every neuron in the next layer (fully connected / Dense layer). This layered structure allows the network to learn more complex, non-linear patterns compared to a single Perceptron.

Key components used:
- **Hidden Layers** – Extra layers between input and output that let the network learn intermediate representations.
- **Activation Functions** – Non-linear functions like **ReLU** (used in hidden layers) and **Sigmoid** (used in the output layer for binary classification) that allow the network to model non-linear relationships.
- **Loss Function** – Measures how far the prediction is from the actual output.
- **Optimizer** – **Adam optimizer** is used to update weights in the direction that reduces the loss, using gradient descent under the hood.
- **Epochs** – The network is trained for 100 epochs, meaning it sees the entire training data 100 times, refining its weights each time.

## Code Concept (Explained Simply)

1. **Dataset** – Same AND gate truth table used in Lab 1: inputs (0,0), (0,1), (1,0), (1,1) with expected outputs 0, 0, 0, 1.

2. **PyTorch Implementation**
   - A custom `FeedforwardNN` class is defined using `nn.Module`.
   - Architecture: Input(2) → Hidden Layer(4, ReLU) → Output Layer(1, Sigmoid).
   - The model is trained manually using a loop: forward pass → calculate BCE loss → backpropagate → update weights with the Adam optimizer.

3. **TensorFlow/Keras Implementation**
   - A `Sequential` model is built with two hidden layers instead of one.
   - Architecture: Input(2) → Hidden Layer(8, ReLU) → Hidden Layer(4, ReLU) → Output Layer(1, Sigmoid).
   - Keras handles training internally via `model.fit()`, which automatically does the forward pass, loss calculation, and backpropagation for each epoch.

4. **Result** – Both models are trained for 100 epochs and then used to predict outputs for all 4 input combinations. Since it's a small dataset trained for a limited number of epochs, the predicted probabilities aren't as sharp as a fully converged model, but the final predicted class matches the AND gate truth table.

## Output

**PyTorch:**
```
Predictions:
Input: [0.0, 0.0] => Predicted: 0.1605 => Class: 0
Input: [0.0, 1.0] => Predicted: 0.3261 => Class: 0
Input: [1.0, 0.0] => Predicted: 0.3353 => Class: 0
Input: [1.0, 1.0] => Predicted: 0.5725 => Class: 1
```

**TensorFlow/Keras:**
```
Predictions:
Input: [0. 0.] => Predicted: 0.4353 => Class: 0
Input: [0. 1.] => Predicted: 0.4845 => Class: 0
Input: [1. 0.] => Predicted: 0.3248 => Class: 0
Input: [1. 1.] => Predicted: 0.5177 => Class: 1
```

## Application

This lab's two hidden layers and dual PyTorch/Keras implementations reflect real, practical choices:

- **Return risk scoring**: combines "item on sale" and "size-return history" — riskier together than either alone.
- **Habit prediction**: an app estimating "risk of skipping tomorrow's workout" from a few interacting signals.
- **Framework choice**: PyTorch's manual loop gives full training control; Keras's `model.fit()` is faster to set up — both common choices for the same kind of network.

The bigger lesson is the epoch comparison with Lab 2: going from 10 to 100 epochs is what fixed the [1,1] misclassification, not the extra layer alone — training time often matters as much as architecture.

## Observations

- Both models correctly classified all 4 cases this time, including **[1,1] as class 1** — the case Lab 2's 10-epoch model got wrong — confirming that the extra 90 epochs (100 total) mattered more than the extra layer.
- PyTorch's predictions were more spread out (0.16 to 0.57) than Keras's (0.43 to 0.52), showing PyTorch's manual training loop reached a more confident decision boundary in the same epoch count.
- Neither model pushed probabilities close to 0 or 1 — correct classification is learned before sharp confidence on this tiny 4-sample dataset.

## Conclusion
The feedforward neural network successfully learned the AND logic using both PyTorch and TensorFlow/Keras, demonstrating how adding hidden layers and non-linear activation functions allows a network to model relationships beyond what a single Perceptron can handle.
