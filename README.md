# Vehicle damage classification with a convolutional neural network

Understanding AI module, MSc AI for Healthcare, University of Hull. First built in January 2026 and rebuilt in August 2026.

A convolutional neural network (CNN), built from scratch in TensorFlow and Keras, that sorts photographs of damaged vehicles into six classes. The rebuilt version reached 0.59 accuracy on 1,440 held-out test images, up from 0.51 in the first version, with a macro-averaged recall of 0.69 across the six classes. The main problem was class imbalance, and the same problem appears in most medical imaging datasets, where the rare class is usually the one that matters.

## The data

7,200 images from the Ripik Hackfest dataset on Kaggle, resized to 128 x 128 pixels, in six classes: crack, dent, glass shatter, lamp broken, scratch and tyre flat. Split 4,608 for training, 1,152 for validation and 1,440 for testing. The classes are heavily imbalanced: crack has 34 test images, scratch has 470.

## What changed in the rebuild

The first version used a learning rate that was too high and did nothing about imbalance, so it leaned towards the most common classes. The rebuild fixed both:

- **Class weights** computed from the class counts, so a mistake on a rare class costs more. Crack received a weight of 6.98, scratch 0.51.
- **Data augmentation** (rotation, shifts, shear, zoom and horizontal flips) to reduce overfitting. Vertical flips were left out, because vehicles are not photographed upside down.
- **A three-block CNN** (32, 64 and 128 filters, batch normalisation, 2.19 million parameters) with early stopping on validation loss.
- **Controlled experiments**, one setting at a time: learning rate (0.01, 0.001 and 0.0001), batch size (32 and 128) and dropout (0.25, 0.5 and 0.75), plus a comparison of regularisation methods.

## Results on the held-out test set

| Class | Precision | Recall | Test images |
|---|---|---|---|
| Glass shatter | 0.72 | 0.94 | 237 |
| Tyre flat | 0.71 | 0.89 | 107 |
| Lamp broken | 0.59 | 0.76 | 176 |
| Crack | 0.12 | 0.71 | 34 |
| Scratch | 0.65 | 0.43 | 470 |
| Dent | 0.68 | 0.42 | 416 |

Overall accuracy 0.59, macro recall 0.69, macro F1 0.58.

**Reading the results.** The class weights did their job on the rare class: crack recall reached 0.71, but at a precision of 0.12, so the model now over-calls cracks. Dents and scratches, the two largest classes, were the hardest, each recalled under half the time. A lower learning rate (0.0001), a batch size of 32 and a dropout of 0.25 gave the best validation accuracy in the experiments.

## Running it

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow
jupyter notebook vehicle_damage_cnn.ipynb
```

Download the Ripik Hackfest vehicle damage dataset from Kaggle and point the configuration cell at the image folder and label CSV. The notebook sorts the images into class folders from the CSV and checks for unreadable files before training.

## What I learned

- Accuracy hides imbalance. A model can look acceptable overall and still miss most of a rare class, so per-class recall comes first.
- Class weights trade precision for recall on rare classes. Whether that trade is right depends on what a missed case costs.
- Change one hyperparameter at a time, or you cannot tell which change helped.

Clovers Akitoye · [GitHub](https://github.com/maryclovers) · [LinkedIn](https://www.linkedin.com/in/cloversakitoye)
