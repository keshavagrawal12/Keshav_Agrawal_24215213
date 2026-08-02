# Lab: Implementing a Binary Classification CNN using PyTorch

## Aim
To implement a CNN using PyTorch and train it to classify images into two categories — Car vs Bike (binary classification).

## Main Topic
Lab 4's MLP worked on tabular data (4 numeric Iris features). Images are different — a 150x150 color image has 67,500 raw pixel values, and a normal MLP can't really pick up on the fact that nearby pixels are related. A CNN fixes this by sliding small filters over the image to detect local patterns (edges, shapes), then stacking layers so later ones combine these into more complex features.

Main components used:
- **Conv2D layers** – learn filters that detect patterns in the image (32 → 64 → 128 filters across 3 blocks, going deeper each time)
- **MaxPooling** – shrinks the feature maps after each conv layer, keeping only the strongest signals
- **AdaptiveAvgPool2d** – forces the output to a fixed 6x6 size so I didn't have to manually calculate the flattened size
- **Fully connected layers** – Linear(512) → Dropout(0.5) → Linear(1), same idea as Lab 4's Dense layers
- **BCEWithLogitsLoss** – binary version of Lab 4's categorical crossentropy, with sigmoid built in
- **Adam optimizer** – same as Lab 4
- **ToTensor()** – scales pixel values 0–255 → 0–1, same purpose as StandardScaler in Lab 4
- **10 epochs** (fewer than Lab 4's 50 since images are heavier to process per batch)

## Code Concept (Explained Simply)
1. **Dataset** – "Car vs Bike Classification Dataset" from Kaggle, ~4,000 images split into `Car` and `Bike` folders. Downloaded automatically using `kagglehub`.
2. **Preprocessing** – `ImageFolder` loads images and auto-labels them from folder names. Images resized to 150x150, split 80/20 into train (3,200) and validation (800).
3. **Model** – Custom `CarBikeCNN` class: 3 Conv+Pool blocks → AdaptiveAvgPool → Linear(512) → Dropout → Linear(1).
4. **Training** – PyTorch doesn't have a `.fit()` like Keras, so the loop is written manually: forward pass → compute loss → `loss.backward()` → `optimizer.step()`, for each batch, each epoch.
5. **Result** – Model evaluated on the validation set, then predictions on a batch converted from probabilities to Car/Bike labels using a 0.5 threshold.

## Output
```
Epoch 1/10  - loss: 0.5485 - accuracy: 0.7278
Epoch 10/10 - loss: 0.0967 - accuracy: 0.9650

Validation Loss: 0.1387
Validation Accuracy: 0.9500

Predicted classes: ['Bike', 'Car', 'Bike', 'Car', 'Bike', 'Car', 'Bike', 'Bike', 'Car', 'Car']
Actual classes:    ['Bike', 'Car', 'Bike', 'Car', 'Bike', 'Car', 'Bike', 'Bike', 'Car', 'Car']
```

## Application
Binary CNNs like this show up anywhere there's a two-class visual decision to make:
- Sorting motorsport footage/photos by vehicle type (cars vs bikes) before doing anything sport-specific
- Medical scans (normal vs abnormal)
- Manufacturing QC (defective vs non-defective)

Same core idea as Lab 4 — learn features, then make a decision — just with conv/pooling layers added at the front to handle image data instead of pre-measured numeric features.

## Observations
- Training accuracy went from 72.78% (epoch 1) to 96.50% (epoch 10), loss dropped from 0.5485 to 0.0967 — steady convergence.
- Validation accuracy ended at 95.00%, close to training accuracy, so not much overfitting — probably because 4,000 images is enough for the model to generalize decently in just 10 epochs.
- Dropout(0.5) likely helped keep that train/val gap small.
- Manually writing the training loop made it a lot clearer what `model.fit()` is actually doing behind the scenes in Keras.

## Conclusion
The CNN classified car vs bike images with 95% validation accuracy after 10 epochs, using PyTorch instead of Keras. Same underlying ideas as Lab 4 (dense layers, matching activation + loss), just with conv/pooling layers added to handle raw image input.
