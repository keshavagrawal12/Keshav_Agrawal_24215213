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
   - Keras handles training internally via `model.fit()`, using `categorical_crossentropy` loss and the Adam optimizer, for 50 epochs with a 10% validation split.

4. **Result** – The trained model is evaluated on the held-out test set and used to predict the species for each test sample. Predictions are converted back from probability vectors to class labels using `np.argmax()`.

## Output

**TensorFlow/Keras:**
```
Test Loss: 0.0971
Test Accuracy: 1.0000

Sample 0: probs=[0.0139, 0.8831, 0.1029] => Predicted class 1 => Actual 1
Sample 1: probs=[0.9897, 0.0099, 0.0004] => Predicted class 0 => Actual 0
Sample 2: probs=[0.0000, 0.0022, 0.9978] => Predicted class 2 => Actual 2
Sample 3: probs=[0.0155, 0.6731, 0.3113] => Predicted class 1 => Actual 1

Predicted classes: [1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0, 1, 2, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0]
Actual classes:    [1, 0, 2, 1, 1, 0, 1, 2, 1, 1, 2, 0, 0, 0, 0, 1, 2, 1, 1, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 0]
```

## Application

Multiclass MLPs like this one are used anywhere a decision has more than two possible outcomes:

- **Species/defect classification**: identifying which of several flower species, product defect types, or disease categories a sample belongs to — the exact structure used here.
- **Digit/character recognition**: classifying an image into one of 10 digits or 26 letters, using the same one-hot + softmax + categorical crossentropy pattern at larger scale.
- **Recommendation tiers**: categorizing a user or transaction into one of several risk/priority tiers instead of a simple yes/no flag.

The bigger lesson connecting back to Lab 3: going from binary to multiclass isn't a new algorithm — it's the same MLP with the output layer, activation, and loss function adapted to handle more than two categories.

## Observations

- The model achieved **100% test accuracy** with a low test loss (0.0971), correctly classifying all 30 held-out samples across the three species.
- Training accuracy rose from **19.4%** (near chance level for 3 classes, ~33%) at epoch 1 to **95.4%** by epoch 50, with validation accuracy following a similar climb from 25% to 91.7%, showing the network steadily learned the decision boundaries.
- Softmax probabilities were sharply confident on easy cases (e.g., Sample 2 at 99.78% for class 2) but noticeably less confident on borderline cases (e.g., Sample 3 at 67.3% vs. 31.1% between classes 1 and 2), reflecting real overlap between the Versicolor and Virginica species in feature space.
- Unlike the AND-gate lab where all four inputs were seen repeatedly, Iris has 150 unique samples, so scaling (`StandardScaler`) played a bigger role here in helping the network converge within 50 epochs.

## Conclusion
The MLP successfully learned to classify Iris flowers into three species using TensorFlow/Keras, demonstrating how a feedforward network handles multiclass problems by combining one-hot encoded labels, a softmax output layer, and categorical crossentropy loss — extending the same core feedforward concepts from binary classification (Lab 3) to a three-class setting.
