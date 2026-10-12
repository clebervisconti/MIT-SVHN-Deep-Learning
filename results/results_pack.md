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
- Training time: 79.3s
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
- Training time: 142.8s
- Accuracy curve: `report_figs/ann2_accuracy_curve.png`

_Pending — Pass 2 not yet run._

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

_Pending — Pass 2 not yet run._

## 5-6. Best-Model Diagnostics

_Pending — Pass 2 not yet run._

## 7. Model Comparison Table

_(see comparison section above once Pass 2 completes)_

## 8. Facts To Check

_Pending — Pass 2 not yet run._
