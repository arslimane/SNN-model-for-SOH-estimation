# 🔋 Lithium-Ion Battery Dataset for Realistic State of Health (SoH) Estimation

Welcome to the official repository for our lithium-ion battery dataset and Spiking Neural Network (SNN)-based SoH estimation models, developed at the **ICube-CNRS Laboratory (INSA Strasbourg)**. This project supports the development of realistic, data-driven battery diagnostics under non-ideal conditions, with a focus on energy-efficient neuromorphic computing.

---

## 📌 Overview

Machine learning (ML) and deep learning (DL) are increasingly used for **State of Health (SoH)** estimation in lithium-ion batteries. However, most existing methods rely on **ideal lab conditions**, limiting their applicability in real-world scenarios.

In this work, we:
- Introduce a **realistic lithium-ion battery dataset**.
- Evaluate a **Spiking Neural Network (SNN)** model for SoH estimation.
- Compare its performance to conventional deep learning methods in terms of **accuracy and energy efficiency**.

---

## 📂 Dataset Description

The dataset includes:
- 🔋 **19 Lithium Iron Phosphate (LFP) cells**
- 🔁 Cycle lifetimes ranging from **500 to 2600 cycles**
- 🌡️ Tested at **three temperatures**: 25°C, 35°C, and 45°C
- 🔄 Includes **non-constant charge/discharge currents**, mimicking real-world usage

### Features
- Time-series voltage, current, temperature data
- Ground truth State of Health (SoH)
- Designed for ML/DL model training and validation

---

## 🧠 Model: Spiking Neural Network (SNN)

We implement and evaluate a brain-inspired **Spiking Neural Network**:
- Based on **event-driven computation**
- Encodes input via **time-coded spikes**
- Offers **temporal precision** and **low power consumption**
- Implemented using `snntorch`

### 🔬 Results
- ✅ **Mean Absolute Error (MAE): 0.01**
- ⚡ Outperforms traditional DL models in **energy efficiency**
- 🧠 Ideal for embedded, **energy-constrained Battery Management Systems (BMS)**



