# Spiking Neural Networks for Accurate and Efficient SOH Estimation of Lithium-Ion Batteries

![Graphical Abstract](Figures/graphicalAbs.pdf)

This repository contains the dataset and code associated with our research on **Spiking Neural Networks (SNNs)** for estimating the **State of Health (SOH)** of lithium-ion batteries across varying temperatures. Developed at **Université de Strasbourg, INSA de Strasbourg, ICube Laboratory (UMR 7357, CNRS)**.

---

## 📖 Overview

Lithium-ion battery health monitoring is essential for ensuring safety, reliability, and optimal performance. Conventional SOH estimation methods often require repeated charge/discharge cycles under strictly controlled laboratory conditions, limiting practical applicability.  

This project provides:

1. **A comprehensive LFP battery dataset**:  
   - 19 lithium iron phosphate (LFP) cells  
   - Cycle lifetimes: 500–2600 cycles  
   - Realistic conditions: **non-constant discharge currents**  
   - Tested at **25°C, 35°C, and 45°C**  

2. **A neuromorphic Spiking Neural Network (SNN) model**:  
   - Mimics biological neurons using **sparse, time-coded spikes**  
   - High temporal precision and **low energy consumption**  
   - Achieves **MAE of 2.61%** for SOH estimation  
   - **Inference time:** 1.09 ms, **Energy consumption:** 0.06 J  

---

## 📊 Results

- **Mean Absolute Error (MAE):** 2.61%  
- **Inference Time:** 1.09 ms  
- **Energy Consumption:** 0.06 J  

The SNN model outperforms conventional deep learning models in computational efficiency while maintaining high predictive accuracy. These characteristics make it particularly suitable for integration into **energy-constrained battery management systems**, supporting both **first-life and second-life applications**.

---

## 📚 Reference

Arbaoui, S., Heitzmann, T., Zitouni, M., et al. *Spiking Neural Networks for Accurate and Efficient State-of-Health Estimation of Lithium-Ion Batteries Across Varying Temperatures*. To be published in **Energy and AI**.
