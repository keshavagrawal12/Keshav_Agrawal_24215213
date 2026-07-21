# Lab 1: Simulating a Perceptron using NumPy

## Aim
To implement a single-layer Perceptron from scratch using NumPy and train it to learn the logical **AND gate**.

## Main Topic

A **Perceptron** is the simplest form of an Artificial Neural Network (ANN), introduced by Frank Rosenblatt. It is a single artificial neuron that takes multiple binary inputs, applies weights to them, sums them up, and passes the result through an **activation function** to produce a single binary output (0 or 1).

It works like a basic decision-maker:
- Each input is multiplied by a **weight** (importance given to that input).
- All weighted inputs are added together along with a **bias**.
- If this sum crosses a certain threshold, the neuron "fires" (outputs 1); otherwise it outputs 0.

A Perceptron can learn simple **linearly separable** problems — meaning problems where a single straight line can separate the outputs into two classes. The **AND gate** is one such example, since a straight line can separate the point (1,1) from the other three points.

## Code Concept (Explained Simply)

1. **Step Function** – Acts as the activation function. If the weighted sum is ≥ 0, output 1; else output 0. This decides whether the neuron fires.

2. **Perceptron Class**
   - `weights` and `bias` start at zero (no knowledge yet).
   - `predict()` calculates the weighted sum (`dot product of inputs and weights + bias`) and passes it through the step function.
   - `train()` repeatedly shows the perceptron all training examples for a number of epochs (rounds). After each prediction, it checks the **error** (difference between actual and predicted output) and updates the weights and bias slightly using the formula:
     ```
     weight = weight + (learning_rate × error × input)
     bias   = bias + (learning_rate × error)
     ```
     This is called the **Perceptron Learning Rule** — it nudges the weights in the right direction whenever the prediction is wrong.

3. **Training Data** – The four possible input combinations for an AND gate (00, 01, 10, 11) with their correct outputs (0, 0, 0, 1).

4. **Result** – After training for 10 epochs, the perceptron correctly learns the AND logic and predicts the exact truth table.

## Output
```
Input: [0 0], Output: 0
Input: [0 1], Output: 0
Input: [1 0], Output: 0
Input: [1 1], Output: 1
```

## Application

The AND gate learned here stands in for any decision that should happen only when **every condition is true at once** — the same pattern shows up in many simple real-world checks:

- **Loan pre-approval**: approve only if "income above limit" AND "debt below limit."
- **Smart irrigation**: turn sprinkler on only if "soil is dry" AND "no rain forecast."
- **Door access**: unlock only if "badge is valid" AND "within working hours."

The real lesson is the *learning process* itself: instead of a programmer hard-coding these AND rules, the Perceptron discovers them on its own from examples — the same basic idea behind real scoring and screening systems that learn from data.

## Observations

- The output above (0, 0, 0, 1) is an **exact match** to the AND truth table — the Perceptron reached 100% accuracy in just 10 epochs over 4 examples.
- Starting from zero weights/bias, it learned the rule purely from the Perceptron Learning Rule updates, with no prior knowledge.
- It succeeds only because AND is linearly separable; a single Perceptron cannot learn non-linearly separable problems like XOR (motivation for Labs 2 and 3).

## Conclusion
The Perceptron successfully learned the AND gate from scratch using only NumPy, showing how a single artificial neuron with weights, bias, and a step activation can solve a simple linearly separable classification problem.