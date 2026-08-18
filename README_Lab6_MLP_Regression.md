# Lab 6: Keras MLP for Regression

## Aim
To implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a regression problem, and analyze how different activation functions and loss functions affect model performance — predicting used-car resale price from listing details.

## Main Topic
Classification answers "which category?" — regression answers "what number?". Predicting a used car's resale price is a real problem every car marketplace (CarDekho, Cars24, Spinny) has to solve: price depends on several numeric and categorical factors at once (how new it is, how much it's been driven, fuel type, who's selling it), and there's no simple formula — it has to be learned from past sale prices. That's exactly what an MLP regressor does: it learns a function that maps a car's features to a continuous price value instead of a class label.

This lab goes a step further than a single model: it systematically compares **3 activation functions** (ReLU, Sigmoid, Tanh) and **3 loss functions** (MSE, MAE, Huber) — 5 total experiments, holding everything else fixed — to see which combination actually works best, rather than assuming the "textbook default" is right.

Main components used:
- **Dense (fully connected) layers** – 64 → 32 → 16 neurons, narrowing down like a funnel
- **Linear output neuron** – the final layer has 1 neuron with no activation, since price can be any real number
- **StandardScaler** – scales all input features to zero mean/unit variance; MLPs converge far better on scaled inputs
- **MSE / MAE / Huber** – the three loss functions compared (see Experiments below)
- **`keras.utils.set_random_seed()` + `enable_op_determinism()`** – needed to get *reproducible* results between runs; without this, TensorFlow gives different numbers each run even with a seed set, which makes a fair comparison impossible
- **Adam optimizer** – same as previous labs
- **One-hot encoding** – converts categorical text columns (Fuel_Type, Seller_Type, Transmission) into numeric columns

## Dataset
**Vehicle Dataset from CarDekho** (Kaggle: `nehalbirla/vehicle-dataset-from-cardekho`), the `car data.csv` file — 301 used-car listings from the Indian resale market, downloaded via `kagglehub`.
- **Target:** `Selling_Price` (resale price, ₹ Lakhs)
- **Features:** `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`, `Year`
- No missing values. `Year` converted to `Car_Age`; `Car_Name` dropped (too many unique values to generalize from). Categoricals one-hot encoded, numerics scaled, split 80/20.

## Understanding the Target Variable (before picking anything)
Before choosing an activation or loss function, it's worth actually looking at what's being predicted:

```
Selling_Price: median = ₹3.6 Lakh, max = ₹35 Lakh, skew = 2.49
```

The most expensive car is **~10x the median** price, and a skew of 2.49 confirms a long right tail — a handful of expensive cars, not a symmetric spread. With only 240 training rows, the model doesn't see many of these expensive cars, so how it's trained to react to large errors on them matters more than it would on a bigger dataset. This is the concrete, data-driven reason behind testing MAE/Huber against MSE in Experiment 2 below — not just "because the textbook says so."

## Why These Activation Functions, For This Setup Specifically
- **Only 240 training rows, 100 fixed epochs** — an activation whose gradient shrinks for large inputs (Sigmoid) has less room to recover from a slow start than one that doesn't (ReLU).
- **Inputs are zero-centered by `StandardScaler`** specifically so the network trains better — ReLU's threshold sits exactly at 0, which lines up with that scaling choice. Sigmoid don't get the same benefit from zero-centering.
- **3 hidden layers deep** — each saturating layer multiplies a small gradient into the next, so the vanishing-gradient risk compounds with depth more than it would in a single-layer network.

Hypothesis going in: **ReLU should train faster and end up more accurate** than Sigmoid here — tested below.

## Why These Loss Functions, For This Dataset Specifically
Given the price skew above (2.49, max ≈10x median), MSE's squared penalty could end up over-weighting the rare expensive cars relative to the much more common cheap ones. MAE and Huber are the standard response to exactly this kind of skew, since neither squares the error.

Hypothesis going in: **MAE or Huber should outperform MSE**, especially on the priciest cars specifically — tested below (see the outlier check).

## MLP Architecture
Input (8 features) → Dense(64) → Dense(32) → Dense(16) → Dense(1, linear). Same architecture reused for every experiment below — only the activation function and loss function change between runs; optimizer, epochs (100), batch size (8), and the train/test split are held fixed.

## Experiment 1 — Activation Function Comparison
Loss held fixed at MSE. Compared ReLU, Sigmoid, Tanh.

| Activation | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **ReLU** | **0.5546** | **0.8352** | **0.9697** |
| Sigmoid | 0.9230 | 1.2624 | 0.9308 |
| Tanh | 0.9127 | 1.5932 | 0.8898 |

ReLU wins clearly — Sigmoid and Tanh's saturating gradients visibly slow convergence in the loss curves (both finish with roughly double ReLU's error).

## Experiment 2 — Loss Function Comparison
Activation held fixed at ReLU (the winner above). Compared MSE, MAE, Huber.

| Loss Function | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **MSE** | **0.5546** | **0.8352** | **0.9697** |
| MAE | 0.6334 | 0.9306 | 0.9624 |
| Huber | 1.0428 | 1.7507 | 0.8669 |

MSE wins overall — its squared-error gradient corrects large mistakes faster within the fixed 100-epoch budget, while MAE/Huber's flatter gradients hadn't fully converged yet.

**But** — checking specifically on the 5 priciest cars in the test set (₹10–23.5 Lakh) flips the story: MAE-loss had the lowest error there (0.911 Lakh) vs. MSE (1.350) and Huber (2.853). MAE's theoretical outlier-robustness does show up — just not enough to overcome MSE's faster overall convergence in this run.

## Comparison Table

| Experiment | Activation | Loss Function | MAE | RMSE | R² |
|---|---|---|---|---|---|
| Model 1 | ReLU | MSE | 0.5546 | 0.8352 | **0.9697** |
| Model 2 | Sigmoid | MSE | 0.9230 | 1.2624 | 0.9308 |
| Model 3 | Tanh | MSE | 0.9127 | 1.5932 | 0.8898 |
| Model 4 | ReLU | MAE | 0.6334 | 0.9306 | 0.9624 |
| Model 5 | ReLU | Huber | 1.0428 | 1.7507 | 0.8669 |

## Analysis & Interpretation
- **Best activation:** ReLU, by a wide margin — cheapest to compute and no vanishing-gradient risk.
- **Best loss overall:** MSE — fastest convergence within the fixed epoch budget.
- **Best combination:** ReLU + MSE (Model 1).
- **Did activation significantly affect convergence?** Yes — Sigmoid and Tanh's validation loss stayed markedly higher throughout training rather than closing the gap with ReLU.
- **Did any loss function help with outliers?** Yes, MAE — lowest error specifically on the 5 priciest cars, consistent with the theory that MAE doesn't let a few large residuals dominate training the way MSE's squared penalty does. Huber underperformed both here, suggesting its default `delta=1.0` wasn't well-calibrated to this price range.
- **Overfitting/underfitting:** No model showed classic overfitting (validation loss climbing while training loss falls). Sigmoid, Tanh, and Huber instead show *underfitting* relative to ReLU+MSE — their validation loss plateaus higher and their training loss hasn't fully converged by epoch 100.

## Final Model Selection
**ReLU + MSE** — best MAE, RMSE, and R² of all 5 experiments, each by a clear margin. Plain MSE beat the "textbook expectation" that MAE/Huber protect better against outliers, at least under this training budget — a reminder that theoretical properties only help once a loss function has actually had room to converge.

```
Final model (ReLU + MSE) - Test set performance:
  MSE:  0.6976
  RMSE: 0.8352
  MAE:  0.5546
  R2:   0.9697

Predictions on the 5 priciest cars in the test set:
  Actual Predicted
   10.11     10.44
   10.90     10.17
   17.00     14.77
   23.00     25.50
   23.50     22.53
```

## Application
MLP regression like this is the backbone of any price/value estimation problem:
- Used vehicle valuation (what this lab does) — the same core idea behind "get an instant quote" tools on car resale platforms
- Real estate price estimation from property features
- Any "estimate the value of X from its attributes" tool — financial valuation done with a neural net instead of a manual formula

## Conclusion
Across 5 systematic experiments, ReLU + MSE was the clear best-performing combination for this dataset and training setup (R² 0.9697), beating both weaker activations (Sigmoid, Tanh) and the "safer" loss functions (MAE, Huber) on every aggregate metric. The one caveat worth remembering: MAE loss specifically outperformed on the priciest cars, a useful reminder that the "best" loss function can depend on which slice of the data — and which epoch budget — you're optimizing for, not just which one wins on average.
