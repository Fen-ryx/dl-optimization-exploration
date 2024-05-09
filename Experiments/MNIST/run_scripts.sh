#!/bin/bash

# Run Python scripts sequentially
python nn_classification.py --opt "SGD" --run-name "SGDMNISTFinal"
python nn_classification.py --opt "SGD-Momentum" --run-name "SGD-MomentumMNISTFinal"
python cnn_classification.py --opt "SGD" --run-name "SGDCNNMNISTFinal"
python cnn_classification.py --opt "SGD-Momentum" --run-name "SGD-MomentumCNNMNISTFinal"
python cnn_classification.py --opt "Adam" --run-name "AdamCNNMNISTFinal"
python cnn_classification.py --opt "AdamW" --run-name "AdamWCNNMNISTFinal"
python cnn_classification.py --opt "RMSProp" --run-name "RMSPropCNNMNISTFinal"
python cnn_classification.py --opt "AdaGrad" --run-name "AdaGradCNNMNISTFinal"