# Lab 2: Building a Neural Network using TensorFlow/Keras

## Aim
To build, train, and evaluate a simple feed-forward Artificial Neural Network (ANN) using TensorFlow/Keras to learn the logical **AND gate**.

## Theory / Main Topic

While Lab 1 used a single Perceptron built manually, this lab uses a **Multi-Layer Neural Network** built with the **Keras** library (part of TensorFlow). A real-world ANN usually has multiple layers of neurons instead of just one, allowing it to learn more complex patterns.

Key building blocks used here:
- **Dense Layer** – A fully connected layer where every neuron receives input from all neurons of the previous layer.
- **Activation Functions**:
  - **ReLU (Rectified Linear Unit)** – Used in the hidden layer; it introduces non-linearity by outputting the input directly if positive, else 0.
  - **Sigmoid** – Used in the output layer; it squashes the output between 0 and 1, ideal for binary classification (yes/no type problems).
- **Loss Function (Binary Crossentropy)** – Measures how far the predicted output is from the actual output, specifically for binary classification tasks.
- **Optimizer (Adam)** – An algorithm that automatically adjusts the weights during training to reduce the loss/error efficiently.
- **Epoch** – One complete pass of the entire training dataset through the network.

Unlike the single Perceptron, this network can, in principle, solve more complex problems because it has a hidden layer that helps capture patterns that aren't linearly separable.

## Code Concept (Explained Simply)

1. **Import Libraries** – TensorFlow (to build/train the network) and NumPy (to handle the data as arrays).

2. **Dataset** – Same AND gate truth table as Lab 1: 4 input pairs and their correct outputs.

3. **Building the Model**
   - `Sequential()` stacks layers one after another.
   - **Hidden Layer**: 4 neurons, takes 2 inputs, uses ReLU activation.
   - **Output Layer**: 1 neuron, uses Sigmoid activation to output a probability between 0 and 1.

4. **Compiling the Model** – Tells Keras *how* to learn: which optimizer to use (Adam), which loss function to minimize (binary crossentropy), and which metric to track (accuracy).

5. **Predictions Before Training** – Since weights are randomly initialized, the model's predictions are essentially random guesses at this stage — this shows the "before learning" state.

6. **Training the Model** – `model.fit()` runs the data through the network for 10 epochs, adjusting weights after each epoch to reduce the loss (using a method called **backpropagation**, handled automatically by Keras).

7. **Predictions After Training** – The model is tested again, and predictions should ideally align closer to the correct AND gate outputs, with accuracy expected to improve with more epochs.

## Output (Sample)
```
Predictions BEFORE training: mostly random values

Predictions AFTER training:
Input: [0. 0.] => Predicted: 0.4975 => Class: 0
Input: [0. 1.] => Predicted: 0.451  => Class: 0
Input: [1. 0.] => Predicted: 0.302  => Class: 0
Input: [1. 1.] => Predicted: 0.2506 => Class: 0
```

*(Note: Accuracy stayed at 75% after 10 epochs — a real neural network typically needs more epochs to fully converge and correctly classify all AND gate cases; this shows how training progressively improves the model.)*

## Conclusion
This lab demonstrates how a multi-layer neural network built with Keras learns from data using layers, activation functions, and an optimizer, contrasting with the manually coded single Perceptron in Lab 1. It also highlights that more epochs/training may be required for the model to fully converge to the correct output.
