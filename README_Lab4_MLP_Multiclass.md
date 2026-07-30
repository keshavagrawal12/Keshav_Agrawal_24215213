# Lab: Implementing a Multiclass Classification MLP using TensorFlow/Keras

## Aim
To implement a Multilayer Perceptron (MLP) using TensorFlow/Keras and train it to classify the **Iris dataset** into its three species (multiclass classification).

## Main Topic

While Lab 3's feedforward network solved a **binary** problem (AND gate: 0 or 1), many real-world problems require choosing between **three or more** classes at once — for example, classifying a flower species, a handwritten digit, or a news article's category. This is called **multiclass classification**, and it requires two key changes to the network compared to a binary MLP:

- **Output layer size** – Instead of a single output neuron, the output layer has one neuron **per class** (3 neurons here, one for each Iris species).
- **Softmax activation** – Instead of Sigmoid (which gives one independent probability), the output layer uses **Softmax**, which converts the raw outputs into a probability distribution across all classes that sums to 1. The class with the highest probability is the model's prediction.

Key components used:
- **One-Hot Encoding** – Class labels (0, 1, 2) are converted into vectors like `[1,0,0]`, `[0,1,0]`, `[0,0,1]` so they match the shape of the softmax output.
- **Hidden Layers** – Two Dense hidden layers (16 and 8 neurons) with **ReLU** activation, allowing the network to learn non-linear boundaries between the three species.
- **Loss Function** – **Categorical Crossentropy**, the multiclass counterpart to binary crossentropy, which measures how far the predicted probability distribution is from the true one-hot label.
- **Optimizer** – **Adam optimizer**, same as Lab 3, used to update weights via gradient descent.
- **Feature Scaling** – `StandardScaler` is applied to the four numeric Iris features (sepal/petal length and width) so they're on a comparable scale before training, which the AND-gate lab didn't need since its inputs were already 0/1.
- **Epochs** – The network is trained for 50 epochs over the scaled Iris data.

## Code Concept (Explained Simply)

1. **Dataset** – The built-in `sklearn` Iris dataset: 150 samples, 4 numeric features (sepal length/width, petal length/width), and 3 target classes (Setosa, Versicolor, Virginica).

2. **Preprocessing**
   - Labels are label-encoded and then one-hot encoded using `to_categorical()`.
   - Data is split 80/20 into train and test sets.
   - Features are standardized using `StandardScaler` (fit on train, applied to test).

3. **TensorFlow/Keras Implementation**
   - A `Sequential` model is built with two hidden layers.
   - Architecture: Input(4) → Hidden Layer(16, ReLU) → Hidden Layer(8, ReLU) → Output Layer(3, Softmax).
   - Total trainable parameters: **243**.
   - Keras handles training internally via `model.fit()`, using `categorical_crossentropy` loss and the Adam optimizer, for 50 epochs with a 10% validation split (batch size 8).

4. **Result** – The trained model is evaluated on the held-out test set and used to predict the species for each test sample. Predictions are converted back from probability vectors to class labels using `np.argmax()`.

## Output

**TensorFlow/Keras:**
```
Epoch 1/50  - accuracy: 0.5000 - loss: 1.3360 - val_accuracy: 0.5000 - val_loss: 1.1886
Epoch 50/50 - accuracy: 0.9352 - loss: 0.1649 - val_accuracy: 0.9167 - val_loss: 0.4147

Test Loss: 0.1253
Test Accuracy: 1.0000

Predicted classes: [1, 0, 2, 1, 1, 0, 1, 2, 1, 1]
Actual classes:    [1, 0, 2, 1, 1, 0, 1, 2, 1, 1]
```
*(Only the first 10 test samples are printed by the code; all 10 shown here match exactly.)*

## Application

Multiclass MLPs like this one are used anywhere a decision has more than two possible outcomes:

- **Species/defect classification**: identifying which of several flower species, product defect types, or disease categories a sample belongs to — the exact structure used here.
- **Digit/character recognition**: classifying an image into one of 10 digits or 26 letters, using the same one-hot + softmax + categorical crossentropy pattern at larger scale.
- **Recommendation tiers**: categorizing a user or transaction into one of several risk/priority tiers instead of a simple yes/no flag.

The bigger lesson connecting back to Lab 3: going from binary to multiclass isn't a new algorithm — it's the same MLP with the output layer, activation, and loss function adapted to handle more than two categories.

## Observations

- The model achieved **100% test accuracy** with a test loss of **0.1253**, correctly classifying all 10 printed held-out samples across the three species (predicted classes matched actual classes exactly).
- Training accuracy started at **50.0%** at epoch 1 (already above the ~33% chance level for 3 classes, since the network quickly separates the easily-distinguishable Setosa class) and rose steadily to **93.52%** by epoch 50. Validation accuracy followed a similar climb, from **50.0%** to **91.67%**.
- Loss dropped consistently across training, from **1.3360** at epoch 1 to **0.1649** at epoch 50 (training) and **1.1886** to **0.4147** (validation), showing smooth, stable convergence with no signs of overfitting (train and validation trends move together).
- Unlike the AND-gate lab where all four inputs were seen repeatedly, Iris has 150 unique samples, so scaling (`StandardScaler`) played a bigger role here in helping the network converge within 50 epochs.

## Conclusion
The MLP successfully learned to classify Iris flowers into three species using TensorFlow/Keras, demonstrating how a feedforward network handles multiclass problems by combining one-hot encoded labels, a softmax output layer, and categorical crossentropy loss — extending the same core feedforward concepts from binary classification (Lab 3) to a three-class setting.