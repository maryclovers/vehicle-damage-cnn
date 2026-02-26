# Vehicle Damage CNN Classifier

MSc AI for Healthcare | University of Hull | Understanding AI Module

## What this is

Built a CNN to classify vehicle damage from photos into 6 categories: crack, scratch, tire flat, dent, glass shatter, lamp broken. The use case is insurance claim verification. Instead of a human manually reviewing every photo, the model flags the damage type automatically.

Dataset is from Kaggle (Ripik Hackfest), 7,200 labelled images.

## What I built

Custom CNN from scratch using TensorFlow/Keras. Three convolutional blocks, each with Conv2D, MaxPooling, and BatchNormalization. Dropout before the final dense layers to reduce overfitting. ReLU throughout, softmax output for the 6 classes.

80/20 train/val split, stratified so each class had proportional representation in both sets. Added data augmentation on the training set (rotation, zoom, flips, shifts) to help the model generalise since some damage classes had fewer images than others.

## Hyperparameter tuning

Tested learning rates, batch sizes, and dropout rates systematically. Learning rate had the biggest impact by far.

Starting with lr=0.01 gave 32% accuracy. Dropping to lr=0.001 jumped it to 51%. Same architecture, same data, totally different result. Dropout rate barely moved the needle between 0.25 and 0.75.

Final model: lr=0.001, batch size=32, dropout=0.5, early stopping with patience=10.

## Results

Validation accuracy: 51% across 6 classes. Random guessing on 6 balanced classes would be 17%, so the model is doing real work. Scratch was the easiest class to detect. Crack was the hardest, frequently confused with other damage types which makes sense visually.

No significant overfitting. Training and validation accuracy tracked closely throughout, gap of less than 1%.

## What I learned

Learning rate matters more than almost anything else at the start. The model was stuck at 32% until I dropped the learning rate. Nothing else changed.

51% on a 6-class problem with limited data and a custom CNN is reasonable, but this is where transfer learning would make a real difference. A pretrained ResNet or EfficientNet would likely push this above 80% because it already knows how to detect edges and textures.

## Tech

Python, TensorFlow/Keras, NumPy, pandas, matplotlib, scikit-learn

## Files

Vehicle_damage_CNN_classifier.ipynb has the full model, training history, confusion matrix, and hyperparameter tuning results.

Dataset not included. Get it from: https://www.kaggle.com/datasets/sudhanshu2198/ripik-hackfest

## Why it matters for healthcare AI

Same architecture applies to medical imaging: chest X-ray classification, skin lesion detection, retinal scans. The challenges are the same too: limited labelled data, class imbalance, need for high recall on rare but serious cases. The difference in healthcare is the cost of a wrong prediction is much higher, so calibration matters more than raw accuracy.

University of Hull, MSc AI for Healthcare, graduating September 2026.
