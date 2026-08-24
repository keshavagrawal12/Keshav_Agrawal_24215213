# Lab 7: Keras MLP for Multiclass Classification

## Aim
To implement a Multi-Layer Perceptron (MLP) using Keras/TensorFlow for a multiclass classification problem, and analyze how different activation functions and optimizers affect classification performance — identifying which class a car is racing in (Hypercar, LMP2 or LMGT3) in the FIA World Endurance Championship from its race data.

## Main Topic
Regression answers "what number?" — classification answers "which category?". The FIA WEC runs three completely different classes of car on the same track at the same time, and all of them are producing lap times, top speeds and pit-stop data all race long. The question a timing screen has to answer is exactly a classification question: given this car's race data, what kind of car is it?

That is harder than it sounds. LMGT3 is obviously slower than everything else, but Hypercar and LMP2 sit only about 3% apart on lap time, and their top speeds overlap almost completely — LMP2 runs low downforce, so at Le Mans it reaches nearly the same speed as the class that laps quicker. This lab compares **3 activation functions** (ReLU, Sigmoid, Tanh) and **3 optimizers** (Adam, SGD, RMSprop) — 5 total experiments, holding everything else fixed — to see which combination handles that overlap best.

Main components used:
- **Dense (fully connected) layers** – 64 → 32 → 16 neurons, same funnel shape as Lab 6
- **Softmax output layer** – 3 neurons, one per class, turning the raw scores into probabilities that add up to 1
- **Sparse categorical cross-entropy** – the classification equivalent of MSE; works directly on integer labels (0/1/2) instead of needing a one-hot target matrix
- **StandardScaler** – the columns run from about 4 pit stops to about 340 kph, so without scaling the big-numbered columns dominate the gradients purely because of their units
- **Stratified train/test split** – keeps the same class proportions in both halves, so the smallest class (LMP2) isn't under-represented in the test set
- **Macro-averaged precision / recall / F1** – treats all three classes equally instead of letting the biggest class decide the score
- **Confusion matrix** – the part that actually shows *where* the model fails
- **Adam / SGD / RMSprop** – the three optimizers compared (see Experiment 2)
- **One-hot encoding** – converts the `Circuit` column into numeric columns

## Dataset
**FIA WEC season race records** — one row is one car in one round of an 8-round season (Lusail, Imola, Spa, Le Mans, Interlagos, COTA, Fuji, Bahrain), giving **424 rows**. The number of cars per class (19 Hypercar, 16 LMP2, 18 LMGT3) and the class performance figures come from the 2024 WEC entry list and the published class regulations; the per-round values are then generated inside those ranges with a fixed seed, so the notebook runs anywhere and gives the same numbers every time, with no Kaggle download or live timing feed needed.
- **Target:** `Class` — Hypercar / LMP2 / LMGT3 (3 classes)
- **Features:** `Circuit`, `Best_Lap_s`, `Avg_Lap_s`, `Top_Speed_kph`, `Laps_Completed`, `Pit_Stops`, `Avg_Stint_Laps`, `Avg_Pit_Time_s`
- No missing values. `Circuit` one-hot encoded, target label-encoded 0/1/2, all features scaled, stratified 80/20 split → 339 train / 85 test, 14 input features.

**Class distribution:** Hypercar 152 (35.8%), LMGT3 144 (34.0%), LMP2 128 (30.2%) — mildly imbalanced, which is why every metric is macro-averaged.

**One deliberate exclusion:** the homologation specs (minimum weight, engine power, hybrid or not) are public but are *not* used as features. Any hybrid car in WEC is a Hypercar by definition, and the class weights — 1030 / 950 / 1300 kg — barely overlap, so including them would give away the answer and turn the task into a lookup table. Using only race data keeps it a real classification problem.

## Understanding the Classes (before picking anything)
Before choosing an activation or an optimizer, it's worth checking how separable the classes actually are. Comparing averages across all 8 circuits mixes up "which class" with "which track", so the comparison that matters is within one circuit:

```
Le Mans only - best lap (s) and top speed (kph) range:
  Hypercar   200.2 - 214.8 s     324.3 - 339.4 kph
  LMP2       211.1 - 219.3 s     325.9 - 339.8 kph
  LMGT3      231.6 - 242.5 s     285.0 - 294.0 kph
```

- **LMGT3 is separated from everything else** — 20+ seconds slower a lap and about 40 kph down on top speed. Any model should get this class easily.
- **Hypercar and LMP2 overlap.** Their lap-time ranges almost touch, and their top speeds overlap *the wrong way round* — both reach about 339 kph, because LMP2 runs low downforce. A model that learns "faster in a straight line = higher class" will get these rows wrong.
- **The circuit has to be an input feature.** A 92 s lap is fast at Interlagos and very slow at Le Mans, so the class can only be read from the lap time *relative to the circuit*.

That last point is why an MLP suits this problem better than a linear model: the boundary is not "lap time below X", it is a different threshold for every circuit — an interaction between the circuit columns and the timing columns, which is exactly what hidden layers are for.

Hypothesis going in: **near-perfect accuracy on LMGT3, with almost all error between Hypercar and LMP2** — tested below.

## Why These Activation Functions, For This Setup Specifically
- **The boundary that matters is a narrow one.** LMGT3 is easy, so nearly all the model's work goes into separating Hypercar from LMP2, about 3% apart. ReLU gives sharp boundaries; Sigmoid and Tanh give smoother, softer ones.
- **Only 339 training rows and 100 fixed epochs** — Sigmoid's gradient is at most 0.25 and shrinks further for large inputs, and three stacked hidden layers multiply those small gradients together, so it starts slowly with limited room to recover.
- **Inputs are zero-centred by `StandardScaler`** — that suits ReLU (its threshold sits at 0) and Tanh (zero-centred output, so the next layer also gets centred inputs). Sigmoid only outputs positive values, so every layer after the first gets shifted, all-positive input.

Hypothesis going in: **ReLU and Tanh should converge quickly and finish close together; Sigmoid should be clearly the slowest** — tested below.

## Why These Optimizers, For This Dataset Specifically
The input is a mix of two very different kinds of column: dense numeric features (lap times, speeds) that change on every row, and sparse one-hot circuit columns that are 0 for 7 rows out of 8. A single global learning rate has to compromise between the two; a per-parameter method does not.

- **SGD (lr 0.01)** — one fixed step size for every weight, the baseline.
- **Adam (lr 0.001)** — tracks both the average and the variance of each weight's gradients and gives every weight its own effective step size.
- **RMSprop (lr 0.001)** — the same per-parameter scaling, without Adam's momentum term.

Hypothesis going in: **the two adaptive optimizers should converge much faster than SGD** and finish close to each other — tested below.

## MLP Architecture
Input (14 features) → Dense(64) → Dense(32) → Dense(16) → Dense(3, softmax). The same architecture is reused for every experiment — only the hidden-layer activation and the optimizer change; loss (sparse categorical cross-entropy), epochs (100), batch size (16), validation split (0.2) and the train/test split are held fixed, and the seed is reset before each model so they all start from the same weights.

## Experiment 1 — Activation Function Comparison
Optimizer held fixed at Adam (lr 0.001). Compared ReLU, Sigmoid, Tanh. All metrics macro-averaged.

| Activation | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **ReLU** | **0.9647** | **0.9638** | **0.9650** | **0.9642** |
| Tanh | 0.9529 | 0.9521 | 0.9521 | 0.9521 |
| Sigmoid | 0.8353 | 0.8333 | 0.8308 | 0.8314 |

Validation accuracy during training:

| Epoch | ReLU | Sigmoid | Tanh |
|---|---|---|---|
| 5 | 0.8088 | 0.5294 | 0.8529 |
| 20 | 0.9118 | 0.6471 | 0.9118 |
| 50 | 0.9118 | 0.8676 | 0.9265 |
| 100 | 0.9265 | 0.8971 | 0.9265 |

Sigmoid needed about 50 epochs to reach the level ReLU and Tanh hit within 5 — exactly the slow start its small gradients predict. Even at epoch 100 it hasn't caught up, and its training loss (0.2034) is still the highest of the three, so it is **underfitting** rather than being wrong in principle. ReLU and Tanh end at the same training accuracy (0.9742), so the gap between them comes down to a single test row.

## Experiment 2 — Optimizer Comparison
Activation held fixed at ReLU. Compared Adam, SGD, RMSprop.

| Optimizer | Test Accuracy | Precision | Recall | F1-Score | Val Acc | Val Loss |
|---|---|---|---|---|---|---|
| **RMSprop** | **0.9647** | **0.9638** | **0.9650** | **0.9642** | **0.9559** | **0.1775** |
| Adam | 0.9647 | 0.9638 | 0.9650 | 0.9642 | 0.9265 | 0.2198 |
| SGD | 0.9059 | 0.9062 | 0.9026 | 0.9037 | 0.9118 | 0.2296 |

Validation accuracy during training:

| Epoch | Adam | SGD | RMSprop |
|---|---|---|---|
| 5 | 0.8088 | 0.5441 | 0.7794 |
| 20 | 0.9118 | 0.7941 | 0.9118 |
| 50 | 0.9118 | 0.8971 | 0.9118 |
| 100 | 0.9265 | 0.9118 | 0.9559 |

Adam and RMSprop tie exactly on the test set, and both are far ahead of SGD on convergence speed — they passed 90% validation accuracy within about 20 epochs, while SGD was still at 0.5441 at epoch 5 and needed around 50 epochs to reach 0.8971. SGD also ends with a lower training accuracy (0.9446 against 0.9742) and its validation loss is still falling at epoch 100, so it is underfitting — it was stopped before it finished learning. Between the two adaptive optimizers, Adam's validation loss bottoms out around epoch 65 and then drifts upward (mild overfitting), while RMSprop's is still at its lowest at epoch 100.

## Confusion Matrix — Where the Errors Actually Are
The best model's confusion matrix (ReLU + RMSprop, 85 test rows):

```
             Predicted
             Hypercar  LMGT3  LMP2
  Hypercar         28      0     2
  LMGT3             0     29     0
  LMP2              1      0    25
```

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Hypercar | 0.9655 | 0.9333 | 0.9492 | 30 |
| LMGT3 | 1.0000 | 1.0000 | 1.0000 | 29 |
| LMP2 | 0.9259 | 0.9615 | 0.9434 | 26 |

- **Most accurate class: LMGT3 — 29/29, perfect on every metric**, and perfect in all five models. Nothing overlaps with it.
- **Hardest pair: Hypercar ↔ LMP2.** Every single error, in every model, is between these two — no prototype was ever confused with a GT car. Even the weakest model (Sigmoid, 14 errors) made all of them in this one pair.
- **Where the errors sit:** 2 of the 3 are at Spa, one of the wet races, where lap times vary far more and the class gap is less clear. The third is a Hypercar at Le Mans with a 336.2 kph top speed — right at the top of the Hypercar range and well inside LMP2 territory. On the features available, that car genuinely looks like an LMP2.

## Comparison Table

| Experiment | Activation | Optimizer | Test Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|
| Model 1 | ReLU | Adam | **0.9647** | **0.9638** | **0.9650** | **0.9642** |
| Model 2 | Sigmoid | Adam | 0.8353 | 0.8333 | 0.8308 | 0.8314 |
| Model 3 | Tanh | Adam | 0.9529 | 0.9521 | 0.9521 | 0.9521 |
| Model 4 | ReLU | SGD | 0.9059 | 0.9062 | 0.9026 | 0.9037 |
| Model 5 | ReLU | RMSprop | **0.9647** | **0.9638** | **0.9650** | **0.9642** |

Training and validation figures for the same five models:

| Experiment | Train Acc | Val Acc | Train Loss | Val Loss |
|---|---|---|---|---|
| Model 1 | 0.9742 | 0.9265 | 0.0583 | 0.2198 |
| Model 2 | 0.9225 | 0.8971 | 0.2034 | 0.2644 |
| Model 3 | 0.9742 | 0.9265 | 0.0637 | 0.2351 |
| Model 4 | 0.9446 | 0.9118 | 0.1397 | 0.2296 |
| Model 5 | 0.9742 | **0.9559** | 0.0641 | **0.1775** |

## Analysis & Interpretation
- **Best activation:** ReLU — 0.9647 test accuracy against 0.9529 for Tanh and 0.8353 for Sigmoid. Its gradients don't saturate, so they stay usable through all three hidden layers.
- **Best optimizer:** RMSprop, just ahead of Adam. Same test scores, but a higher validation accuracy (0.9559 vs 0.9265) and lower validation loss (0.1775 vs 0.2198).
- **Best combination:** ReLU + RMSprop (Model 5).
- **Did activation significantly affect convergence?** Yes — Sigmoid was at 0.5294 validation accuracy at epoch 5 against 0.8088 (ReLU) and 0.8529 (Tanh), and took about 50 epochs to catch up to where they started.
- **Did the optimizer affect convergence speed?** Yes, more than the activation did. Adam and RMSprop passed 90% within about 20 epochs; SGD took around 50 to reach 0.8971.
- **Train vs test gap:** small everywhere. Models 1 and 5 reach 0.9742 training accuracy against 0.9647 on test — under 1 point. Sigmoid scores higher on validation (0.8971) than test (0.8353), which just reflects the small validation set (68 rows) rather than anything meaningful.
- **Overfitting/underfitting:** mostly underfitting for the weaker models. Sigmoid and SGD both still have falling training loss at epoch 100 and their validation loss is lowest at the final epoch, so they were stopped before they finished learning. Adam shows mild overfitting — validation loss bottoms out around epoch 65 then drifts up. RMSprop is the most stable of the five.
- **Which classes were misclassified, and why?** Only Hypercar and LMP2. Their lap times are about 3% apart and their top speeds overlap almost entirely because LMP2 runs low drag. Balance of Performance deliberately pushes the classes toward each other, the wet races add extra variation in lap time, and within-class spread is large — a struggling Hypercar and a strong LMP2 look nearly identical in the data.
- **What would improve it:** early stopping on validation loss (Adam peaked around epoch 65), more epochs for Sigmoid and SGD which were both still improving, better features for the hard pair (sector times, or the gap to the fastest lap of the round instead of raw top speed, which is actively misleading here), more seasons of data, and class weights for the smaller LMP2 class.

## Final Model Selection
**ReLU + RMSprop** (Model 5) — joint-best test accuracy (0.9647) and macro F1 (0.9642), plus the best validation accuracy (0.9559) and lowest validation loss (0.1775) of all five models. Per class, LMGT3 is perfect and even the weakest class (LMP2) is above 0.94 F1. Only 3 of 85 test rows are wrong, all between Hypercar and LMP2, and all on genuinely ambiguous rows — 2 from the wet race at Spa and 1 from a Hypercar running LMP2-level top speed at Le Mans. Model complexity is identical to every other model here, so nothing is being bought with extra parameters.

The honest caveat: Model 1 (ReLU + Adam) matches it exactly on the test set, and the difference is 3 rows either way. The tiebreaker is the validation evidence — RMSprop's validation loss was still improving at epoch 100, while Adam's had started drifting upward.

## Application
Multiclass MLP classification like this shows up wherever a category has to be worked out from measured performance rather than looked up:
- Race control and timing systems — flagging a car whose performance no longer matches its declared class, which is how Balance of Performance breaches get spotted
- Motorsport data analysis — grouping stints or laps by car type from timing data alone
- Any "which category does this belong to, given its measurements" problem — equipment classification, quality grading, tier assignment

## Conclusion
Across 5 systematic experiments on the same split and the same 64-32-16 architecture, **ReLU + RMSprop** was the best combination for this dataset (96.47% test accuracy, 0.9642 macro F1). The results split exactly along the lines the data predicted: LMGT3 was classified perfectly by every model, and every error in every model was between Hypercar and LMP2, concentrated at the wet race and at Le Mans where both classes reach nearly the same top speed.

That remaining error isn't really a modelling failure. Balance of Performance exists specifically to push these two classes toward each other, so their race data is genuinely similar and some rows cannot be separated from the features available. The activation and optimizer results behaved as the theory predicts — Sigmoid's saturating gradients made it by far the slowest to learn and left it underfitting at epoch 100, and SGD had the same problem for a different reason: one fixed step size cannot suit both the dense timing columns and the sparse one-hot circuit columns at once, which is exactly the problem adaptive optimizers solve.
