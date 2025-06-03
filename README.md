# Fast-Ion Flow Event Detection in the Earth's Magnetosphere

This project addresses the detection of fast-ion plasma flow events in the Earth's nightside magnetosphere using high-frequency satellite magnetic field data. The focus is on building a volatility-informed event detection pipeline that leverages both domain-specific statistical features and deep learning techniques.

---

## Overview

Fast-ion flow events are critical to understanding magnetospheric dynamics and substorm activity. However, due to their rarity and the high-frequency nature of the underlying satellite data, identifying these events in real time remains a challenge.

This project proposes a scalable, data-driven approach for detecting fast-ion flow intervals by analyzing volatility patterns in magnetic field fluctuations. The resulting model demonstrates strong event capture rates and generalization across different satellites.

---

## Data Source

- **Mission**: NASA THEMIS (Time History of Events and Macroscale Interactions during Substorms)
- **Satellites Used**: Themis A, D, and E
- **Sampling Rate**: 3-second frequency
- **Data Volume**: ~60 million rows of magnetic field measurements
- **Features Used**:
  - $\( B_x, B_y, B_z \)$ magnetic field components
  - Derived volatility features:
    - AR(1)-GARCH(1,1) volatility estimates per component
    - 9-window rolling standard deviations per component

All features were scaled to [0, 1] using min-max normalization per satellite. Labels for fast-ion flow events were constructed based on velocity thresholds, applied independently per satellite.

- Link for cleaned data: [Box Link](https://utdallas.box.com/s/jufvj71jfnmu7sn5fm2wgtee3o48i3zc)

---

## Motivation

Traditional event detection models often rely on hand-tuned thresholds or statistical baselines that fail under dynamic plasma conditions. Our goal was to build a model that could learn temporal volatility patterns associated with fast-ion flows—without directly using velocity as an input.

Volatility was chosen as the modeling basis because:
- It reflects local plasma instability and turbulence
- It exhibits precursor signatures prior to fast-ion events
- It provides richer temporal dynamics than raw magnitude

---

## Modeling Approach

A hybrid approach was used that combines physics-aware statistical feature engineering with deep sequence modeling:

- **Feature Engineering**:
  - Volatility modeled using AR(1)-GARCH(1,1) to capture memory effects. The AR(1)-GARCH(1,1) can be interpreted as a stochastic difference equation – a discrete-time analog to stochastic differential equations. This allows us to estimate both the drift and diffusion behavior of the magnetic field time series.
  - Localized fluctuation captured via 9-timestep rolling standard deviations

- **Neural Network Architecture**:
  A custom deep learning model, **TimeSeqNet**, was designed to detect sequences of volatility patterns preceding fast-ion events.
  - Convolutional layer for local pattern extraction
  - Bidirectional LSTM layer for temporal context
  - Multi-head attention for per-timestep interpretability
  - Sequence-to-sequence output structure with aggregation at the interval level
  - Multi-Output Shared Representation Learning to capture per timestep information and latent feature representations for event sequences 
  - Trained using a custom loss function with Binary Crossentropy and Tversky loss to address class imbalance

The model takes in sequences of engineered features and outputs per-timestep probabilities, which are aggregated to produce a final event classification for each interval.

---

## Evaluation Strategy

- **Training/Test Splits**:
  - Satellite-randomized and time-respecting splits were used
  - Additional generalization testing: train on A+D, test on E

- **Metrics**:
  - Precision, recall, and F1 score computed at multiple thresholds
  - Event-level evaluation using captured vs. missed intervals
  - Missed to Captured Events ratio over time
  - Data drift analysis

- **Best Performance (10 Epochs, F1-Optimal Threshold = 0.58)**:
  - Precision: 0.76
  - Recall: 0.83
  - F1 Score: 0.79

The model was found to generalize well across satellites and years, with consistent recall and stability over time.

---

## Summary

This project demonstrates that volatility-driven modeling is an effective strategy for detecting fast-ion plasma flow events in the magnetosphere. By leveraging GARCH and rolling standard deviation features alongside a custom sequence learning architecture, we achieve robust, generalizable event detection on high-frequency satellite data.

---

