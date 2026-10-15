# Lab 8 – RNN-Based Text Classification on Formula 1 Fan Tweets

## Aim
To implement a Recurrent Neural Network (SimpleRNN) in TensorFlow/Keras that classifies Formula 1 fan tweets as Positive (1) or Negative (0), study how RNN units, epochs and batch size affect performance, and compare a single-layer and a two-layer RNN. A motorsport dataset is used instead of IMDB, with the teacher's permission.

## Main Topic
Recurrent Neural Networks for text classification: Embedding → SimpleRNN → Dense (sigmoid), trained with Adam and binary cross-entropy.

## Components
| Component | What it does |
|---|---|
| Tweet loader + cleaning | Reads `F1_tweets.csv`, removes links, @mentions and `#`, drops duplicates, retweets and non-English tweets |
| VADER labelling | Gives each tweet a compound score; ≥ +0.3 = Positive, ≤ −0.3 = Negative, the rest are dropped |
| Vocabulary (10,000 words) | Built from training tweets only; id 0 = padding, id 1 = unknown word |
| `pad_sequences(maxlen=50)` | Pads short tweets with zeros and truncates long ones so every tweet has 50 tokens |
| `Embedding(10000, 128)` | Turns each word id into a learned 128-dimensional vector |
| `SimpleRNN(64)` | Reads the 50 vectors in order and keeps a hidden state; the last state summarises the tweet |
| `Dense(1, sigmoid)` | Outputs the probability that the tweet is positive |
| `return_sequences=True` | Makes an RNN layer return its hidden state at every step so a second RNN layer can follow |

## Dataset
Real Formula 1 tweets from the 2021–2022 seasons (Mexico City, Qatar, British GP, Abu Dhabi and others), from the public GitHub repository `NissyAbrahamA/Sentiment-Analysis` (`F1/F1_tweets.csv`).

| Step | Tweets left |
|---|---|
| Tweet records in the file | 131,365 |
| English, unique, no retweets | 85,137 |
| Clear sentiment (|VADER score| ≥ 0.3) | 52,557 (39,011 positive, 13,546 negative) |
| Balanced by down-sampling positives | 27,092 (13,546 per class) |
| Train / test (80 / 20, stratified) | 21,673 / 5,419 |

Final shapes: `x_train (21673, 50)`, `x_test (5419, 50)`.

**Important about the labels.** The file has no sentiment labels, so they come from the VADER lexicon, not from people. The model therefore learns to reproduce VADER's judgement from the word sequence, and "test accuracy" means agreement with VADER. The notebook prints six random tweets per class so the labels can be checked by eye. The positive class is down-sampled because fans post far more praise than complaints (74% positive), and an unbalanced set would let the model score well by guessing "positive".

## Why these choices
- **Sigmoid + binary cross-entropy:** two classes, one probability output; the loss punishes confident wrong answers heavily.
- **Adam:** adaptive learning rate, a safe default for RNNs.
- **Pre-padding:** zeros at the start, so the last words of a tweet sit next to the RNN's final hidden state.
- **MAX_LEN = 50:** tweets average about 26 words and only 2.8% are longer than 50.
- **Seeds reset before every model**, so differences between experiments come from the hyperparameter and not from random initialisation.

## Architecture
| Layer | Output shape | Parameters |
|---|---|---|
| Embedding | (None, 50, 128) | 1,280,000 |
| SimpleRNN (64) | (None, 64) | 12,352 |
| Dense (1, sigmoid) | (None, 1) | 65 |

Total: 1,292,417 parameters. The embedding table is almost all of it.

## Base model result (64 units, 5 epochs, batch 64)
| Epoch | Train Acc | Val Acc | Val Loss |
|---|---|---|---|
| 1 | 0.6780 | 0.8189 | 0.4233 |
| 2 | 0.8998 | 0.8471 | 0.3899 |
| 3 | 0.9589 | 0.8205 | 0.6069 |
| 4 | 0.9718 | 0.8035 | 0.5852 |
| 5 | 0.9867 | 0.7319 | 0.8465 |

Test loss 0.8005, **test accuracy 73.56%**. Training accuracy keeps rising while validation loss is lowest at epoch 2 and then climbs: the model is memorising the training tweets. The last epoch also happens to be an unstable one (validation accuracy dropped about 7 points in a single epoch), which is why the base setting looks weaker than most other runs below.

## Experiments
Base setting: 64 units, 5 epochs, batch size 64, validation split 20%. One hyperparameter changes at a time, so A2, B2 and C2 are the same run.

| Experiment | Parameter | Training Acc | Validation Acc | Test Acc | Time (s) |
|---|---|---|---|---|---|
| A1 | RNN = 32 | 0.9852 | 0.8009 | 0.8164 | 35 |
| A2 | RNN = 64 | 0.9867 | 0.7319 | 0.7356 | 41 |
| A3 | RNN = 128 | 0.9737 | 0.7968 | 0.8040 | 49 |
| B1 | Epochs = 3 | 0.9589 | 0.8205 | 0.8286 | 23 |
| B2 | Epochs = 5 | 0.9867 | 0.7319 | 0.7356 | 40 |
| B3 | Epochs = 10 | 0.9969 | 0.7979 | 0.8046 | 81 |
| C1 | Batch = 32 | 0.9847 | 0.8002 | 0.8022 | 86 |
| C2 | Batch = 64 | 0.9867 | 0.7319 | 0.7356 | 45 |
| C3 | Batch = 128 | 0.9896 | 0.8270 | 0.8389 | 23 |

(Times vary a little between runs on a CPU; accuracies are reproducible with the fixed seeds on the same TensorFlow version.)

**What the experiments show**
- **Units:** 32 units (81.6%) did as well as or better than 64 and 128. More units gave no gain, and 128 units took longer per epoch.
- **Epochs:** 3 epochs (82.9%) beat 5 and 10. Training accuracy goes up with more epochs (0.96 → 0.997) but test accuracy does not; validation loss is best around epoch 2. This is overfitting, and early stopping would be the fix.
- **Batch size:** batch 128 gave the best test accuracy (83.9%) and the shortest time; batch 32 was four times slower with no benefit.
- **Reliability:** the spread between runs (73.6% to 83.9%) is mostly last-epoch instability of the SimpleRNN, not a real effect of the hyperparameter, so differences of a few points should not be over-read.

## Single-layer vs two-layer RNN
`Embedding → SimpleRNN(64, return_sequences=True) → SimpleRNN(64) → Dense`. The first layer must return the hidden state of every time step, shape `(batch, 50, 64)`, because the second RNN needs a sequence as input.

| | Single-layer | Two-layer |
|---|---|---|
| Training accuracy | 0.9867 | 0.9863 |
| Validation accuracy | 0.7319 | 0.8092 |
| Test accuracy | 0.7356 | 0.8245 |
| Training time | about 40 s | about 51 s |
| Parameters | 1,292,417 | 1,300,673 |

The two-layer model scored higher and trained longer. But the single-layer number is the unstable base run; the average of all nine single-layer runs is 78.9%, so the honest reading is "two-layer is probably a little better, not proven by one run". More layers do not always help a SimpleRNN.

## Text prediction (Task 8)
`predict_sentiment(text)` cleans the text, maps words to the training vocabulary, pads to 50 and returns the label with its probability. Five F1 fan comments written by me (not in the dataset) were all classified correctly (5/5), with probabilities between 0.0001 and 0.9962.

## Motorsport application
A series, team or broadcaster receives thousands of fan posts after a race. A sentiment classifier gives a first sort into positive and negative so the media desk reads the complaints first. As a domain-shift check, ten WEC-style comments (Fuji, Le Mans, Bahrain) written by me were all classified correctly (10/10). These comments use plain sentiment words, which is why it works; comments where the sentiment sits in racing terms (undercut, stint, BoP) would be harder, and a real system would need human-labelled WEC comments.

## Discussion questions
All ten are answered in the notebook (section 13): RNN and why it suits text, Embedding, padding, sigmoid, binary cross-entropy, vanishing gradient, effect of more units, overfitting, reducing overfitting, SimpleRNN vs LSTM vs GRU.

## Final Model Selection
Chosen on **validation** accuracy so the test set stays an honest final check: **C3 (64 units, 5 epochs, batch 128)**, validation 0.8270, test 0.8389.

## Conclusion
A SimpleRNN with an Embedding layer can learn the sentiment of F1 fan tweets to about 80–84% agreement (the unstable base run reached 73.6%) with the VADER labels. It overfits quickly (training accuracy near 99%), the best validation point is around epoch 2, and more units or more epochs did not improve test accuracy. Batch size 128 gave the best and fastest run. The main limits are the unstable SimpleRNN training (vanishing gradients) and labels that come from a lexicon instead of people. An LSTM or GRU with dropout and early stopping would give steadier results.

## How to run
1. Open `rnn_lab8_motorsport.ipynb` in Google Colab (it clones the dataset from GitHub by itself) or locally with `F1_tweets.csv` next to it.
2. Runtime → Run all (CPU is fine, about 10 minutes).
3. If your numbers differ a little because of a different TensorFlow version, use your own numbers in the report.
