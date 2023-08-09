# AI Shape Detector

This project is a Python script that uses Keras and TensorFlow to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for shape detection in images. The trained model is then evaluated on the test set and saved to the project directory.

## Getting Started

To use this project, you will need to have Keras and TensorFlow installed in your Python environment. 

## Prerequisites

- Python 3.x
- TensorFlow
- Keras

## Installation

1. Clone the repository to your local machine.
2. Ensure that you have the necessary Python packages installed (TensorFlow and Keras).

## Usage

To run the script, simply execute the script in a Python environment. The script will load and normalize the CIFAR-10 dataset, define a CNN model, compile the model with an Adam optimizer and a Sparse Categorical Crossentropy loss function, train the model on the training set, evaluate the model on the test set, and save the trained model to the project directory. The model is specifically designed to detect shapes in images.

## Built With

- Python
- TensorFlow
- Keras

## Acknowledgments

- The CIFAR-10 dataset was used for training and testing the model.
- The model architecture includes convolutional layers for feature extraction and dense layers for classification. These layers are particularly effective for shape detection in images.

## Contact

* [Kishan Rajeev](https://kishan.knowledgeplatter.com/)

## License

Planr Pro is licensed under the MIT License.
