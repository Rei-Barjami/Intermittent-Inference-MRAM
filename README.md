# Project Overview

This repository contains the codebase used for the evaluation presented in our paper. Below is a brief description of each file and its purpose.

## Folder Structure and File Descriptions

- **`models/`**  
  This folder contains the pre-trained models used in our evaluation experiments.

- **`errorInj.py`**  
  Implements our custom error injection layer, which simulates faults in the neural network computation. This layer is used to evaluate model robustness under various error conditions.

- **`dataset_p.py`**  
  Handles preprocessing of the STM Flowers dataset. This preprocessing is applied to all image classification models evaluated in the paper.

- **`model_set.py`**  
  Contains utilities to load and quantize the models. It returns the quantized version of a given model, ready for evaluation.

- **`enConComp.py`**  
  Computes the energy consumption for different quality levels of STT-MRAM. It requires as input the energy consumption data of running the model on an MCU and estimates the corresponding memory energy costs.

- **`accuracyCalc.py`**  
  Evaluates and prints the classification accuracy of the model at each quality level, allowing comparison across different error or memory conditions.

- **`mainTest.py`**  
  Provides a complete example of how to run both accuracy and energy consumption evaluations. By default, it runs the analysis on the `fd_mobilenet_0.25_128` model using the Flowers dataset.


## Requirements

- Python >= 3.8
- tensorflow==2.15.0
- tensorflow_model_optimization==0.7.5
- NumPy

## How to Run

```bash
python mainTest.py
