# Deep Learning for Probability Estimation

A practical machine learning project exploring how a neural network can approximate a computationally expensive mathematical function. The work covers synthetic data generation, feature preprocessing, supervised learning, model evaluation, and analysis of the tradeoff between accuracy and inference speed.

## Workflow

1. **Problem analysis and target definition.** Studied the reference function and formulated its approximation as a supervised learning problem with continuous probability targets.

2. **Synthetic dataset generation.** Generated 3,200 labeled examples with varying input lengths and feature values. Used the reference calculation to produce ground-truth targets, keeping the supplied prediction dataset separate from training and validation.

3. **Data preprocessing.** Used NumPy and pandas to organize the data, normalize feature scales, pad variable-length inputs, and construct training arrays. Shuffled and split the generated dataset into training and validation subsets.

4. **Deep learning model development.** Built a fully connected neural network using TensorFlow/Keras, with ReLU hidden layers, dropout regularization, and softmax outputs representing a probability distribution.

5. **Training and optimization.** Trained with the Adam optimizer and mean squared error loss. Applied learning-rate reduction and early stopping based on validation mean absolute error to control training duration and restore selected model weights.

6. **Data analysis and evaluation.** Visualized training and validation MSE and MAE using Matplotlib, examined learning curves, and compared sample predictions with reference values. Benchmarked warmed, compiled inference against the original calculation to assess computational efficiency.

7. **Prediction and reporting.** Applied consistent preprocessing and output postprocessing to new examples, exported predictions to CSV, and documented the methods and results in an executed notebook and HTML report.

The project demonstrates an end-to-end workflow for neural function approximation, from creating labeled data to evaluating and communicating model results.

**Tools:** Python, TensorFlow/Keras, NumPy, pandas, Matplotlib.

[Notebook](project1.ipynb) · [HTML report](project1.html) · [Predictions](results.csv)
