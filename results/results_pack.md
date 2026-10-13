# SVHN Digit Recognition — Results Pack

Device used for training: `GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

Seed: 42 (NumPy, Python random, TensorFlow/Keras). Note: Metal GPU execution is not guaranteed to be bit-for-bit deterministic even with fixed seeds.

## 1. Data Overview

- Train shape: (42000, 32, 32), dtype float32
- Test shape: (18000, 32, 32), dtype float32
- Image size: 32x32 grayscale
- Classes: 10 (digits 0-9)
- Sample image grid: `report_figs/sample_images.png`
- Class distribution bar chart: `report_figs/class_distribution.png`
- Train class counts: 0:4186, 1:4172, 2:4197, 3:4281, 4:4188, 5:4232, 6:4168, 7:4192, 8:4188, 9:4196
- Test class counts: 0:1814, 1:1828, 2:1803, 3:1719, 4:1812, 5:1768, 6:1832, 7:1808, 8:1812, 9:1804

## 2. Model Performance

### ANN-1

```
Model: "ANN_1_baseline"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 256)            │       262,400 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 304,202 (1.16 MB)
 Trainable params: 304,202 (1.16 MB)
 Non-trainable params: 0 (0.00 B)

```
- Parameters: 304,202
- Train accuracy: 10.20%
- Validation accuracy: 10.67%
- Test accuracy: 10.47%
- Epochs: 20
- Training time: 73.6s
- Accuracy curve: `report_figs/ann1_accuracy_curve.png`

### ANN-2

```
Model: "ANN_2_deeper"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_12 (Dense)                │ (None, 512)            │       524,800 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 512)            │         2,048 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu (LeakyReLU)         │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (Dense)                │ (None, 256)            │       131,328 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 256)            │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_1 (LeakyReLU)       │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (Dense)                │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 128)            │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_2 (LeakyReLU)       │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_15 (Dense)                │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_3 (LeakyReLU)       │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_16 (Dense)                │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 701,514 (2.68 MB)
 Trainable params: 699,722 (2.67 MB)
 Non-trainable params: 1,792 (7.00 KB)

```
- Parameters: 701,514
- Train accuracy: 83.85%
- Validation accuracy: 68.89%
- Test accuracy: 68.77%
- Epochs: 20
- Training time: 131.9s
- Accuracy curve: `report_figs/ann2_accuracy_curve.png`

### CNN-1

```
Model: "CNN_1_baseline"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 30, 30, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 15, 15, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 13, 13, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 2304)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_27 (Dense)                │ (None, 64)             │       147,520 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_28 (Dense)                │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 166,986 (652.29 KB)
 Trainable params: 166,986 (652.29 KB)
 Non-trainable params: 0 (0.00 B)

```
- Parameters: 166,986
- Train accuracy: 72.04%
- Validation accuracy: 73.51%
- Test accuracy: 73.38%
- Epochs: 20
- Training time: 92.6s
- Accuracy curve: `report_figs/cnn1_accuracy_curve.png`

### CNN-2

```
Model: "CNN_2_deeper"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_6 (Conv2D)               │ (None, 30, 30, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_12 (LeakyReLU)      │ (None, 30, 30, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_9           │ (None, 30, 30, 32)     │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_7 (Conv2D)               │ (None, 28, 28, 32)     │         9,248 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_13 (LeakyReLU)      │ (None, 28, 28, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_6 (MaxPooling2D)  │ (None, 14, 14, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_9 (Dropout)             │ (None, 14, 14, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_8 (Conv2D)               │ (None, 12, 12, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_14 (LeakyReLU)      │ (None, 12, 12, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_10          │ (None, 12, 12, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_9 (Conv2D)               │ (None, 10, 10, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_15 (LeakyReLU)      │ (None, 10, 10, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_7 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_10 (Dropout)            │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_3 (Flatten)             │ (None, 1600)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_33 (Dense)                │ (None, 128)            │       204,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_16 (LeakyReLU)      │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_11 (Dropout)            │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_34 (Dense)                │ (None, 10)             │         1,290 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 271,594 (1.04 MB)
 Trainable params: 271,402 (1.04 MB)
 Non-trainable params: 192 (768.00 B)

```
- Parameters: 271,594
- Train accuracy: 94.73%
- Validation accuracy: 92.89%
- Test accuracy: 93.06%
- Epochs: 20
- Training time: 159.3s
- Accuracy curve: `report_figs/cnn2_accuracy_curve.png`

## 3. Main Differences

### ANN-1 vs ANN-2

- ANN-1 layers: 3 Dense (256, 128, 64) + output, all ReLU
- ANN-2 layers: 4 Dense (512, 256, 128, 64) + output, LeakyReLU(0.1)
- ANN-1 Dropout: none
- ANN-2 Dropout: Dropout(0.2) after first three Dense blocks
- ANN-1 BatchNorm: none
- ANN-2 BatchNorm: after Dense(512), Dense(256), Dense(128)
- ANN-1 activation: ReLU
- ANN-2 activation: LeakyReLU(negative_slope=0.1)
- ANN-1 parameter count: 304202
- ANN-2 parameter count: 701514

### CNN-1 vs CNN-2

- CNN-1 layers: 2 Conv2D (32, 64) + MaxPool each + Dense(64) + output, ReLU
- CNN-2 layers: 4 Conv2D (32, 32, 64, 64) + MaxPool x2 + Dense(128) + output, LeakyReLU(0.1)
- CNN-1 Dropout: none
- CNN-2 Dropout: Dropout(0.25) x2 (after each pooling block) + Dropout(0.4) before output
- CNN-1 BatchNorm: none
- CNN-2 BatchNorm: after first Conv2D in each of the two conv blocks
- CNN-1 activation: ReLU
- CNN-2 activation: LeakyReLU(negative_slope=0.1)
- CNN-1 parameter count: 166986
- CNN-2 parameter count: 271594

## 5-6. Best-Model Diagnostics

### Best ANN: ANN-2

- Confusion matrix: `report_figs/confusion_matrix_ann_best.png`
- Classification report image: `report_figs/classification_report_ann_best.png`
```
              precision    recall  f1-score   support

           0     0.7564    0.8456    0.7985      1814
           1     0.8111    0.6625    0.7293      1828
           2     0.9503    0.5829    0.7226      1803
           3     0.7154    0.5980    0.6515      1719
           4     0.9741    0.5397    0.6946      1812
           5     0.3555    0.9135    0.5118      1768
           6     0.7740    0.6861    0.7274      1832
           7     0.7770    0.8191    0.7975      1808
           8     0.8640    0.5397    0.6644      1812
           9     0.7209    0.6901    0.7052      1804

    accuracy                         0.6877     18000
   macro avg     0.7699    0.6877    0.7003     18000
weighted avg     0.7711    0.6877    0.7010     18000

```
- Top 5 most-confused (true -> predicted, count) pairs: [(3, 5, 535), (8, 5, 441), (4, 5, 437), (6, 5, 423), (1, 5, 311)]

### Best CNN: CNN-2

- Confusion matrix: `report_figs/confusion_matrix_cnn_best.png`
- Classification report image: `report_figs/classification_report_cnn_best.png`
```
              precision    recall  f1-score   support

           0     0.9299    0.9587    0.9441      1814
           1     0.8649    0.9524    0.9065      1828
           2     0.9482    0.9434    0.9458      1803
           3     0.9178    0.9098    0.9138      1719
           4     0.9477    0.9393    0.9435      1812
           5     0.9127    0.9400    0.9262      1768
           6     0.9436    0.9039    0.9233      1832
           7     0.9730    0.9159    0.9436      1808
           8     0.9272    0.9272    0.9272      1812
           9     0.9515    0.9141    0.9324      1804

    accuracy                         0.9306     18000
   macro avg     0.9316    0.9305    0.9306     18000
weighted avg     0.9317    0.9306    0.9307     18000

```
- Top 5 most-confused (true -> predicted, count) pairs: [(7, 1, 81), (3, 5, 53), (6, 5, 48), (6, 8, 47), (4, 1, 46)]

## 7. Model Comparison Table

```
Model  Parameters  Train Acc  Val Acc  Test Acc  Epochs  Train Time (s)
ANN-1      304202   0.101964 0.106667  0.104667      20            73.6
ANN-2      701514   0.838542 0.688929  0.687667      20           131.9
CNN-1      166986   0.720446 0.735119  0.733833      20            92.6
CNN-2      271594   0.947262 0.928929  0.930556      20           159.3
```

## 8. Facts To Check

- Label range confirmed as 0-9 for both train and test (np.unique output above).
- Pixel scaling applied: True (raw max was > 1, so divided by 255).
- Device used for training: GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')].
- Validation definition: validation_split=0.2 of the training array at fit() time (not the shipped X_val/y_val arrays in the .h5 file).
- Seed used: 42 for Python random, NumPy, and TensorFlow/Keras (keras.utils.set_random_seed). Metal GPU execution is not guaranteed bit-for-bit deterministic even with fixed seeds.
- Epochs completed per model: ANN-1=20, ANN-2=20, CNN-1=20, CNN-2=20 (no early stopping used).
- Best-ANN selection rule: highest test accuracy, then validation accuracy, then fewer parameters. Selected: ANN-2.
- Best-CNN selection rule: same rule. Selected: CNN-2.
- Batch size: 128 for all four models.
- The .h5 file also ships an unused X_val/y_val split (60,000 images); it is not used anywhere in this notebook.
