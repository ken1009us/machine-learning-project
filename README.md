# Text Classification using GRU in Keras

## Description
```
- This code implements a text classification model using GRU (Gated Recurrent Unit) in Keras.
- The model is trained on a dataset containing text samples and their corresponding categories.
- The goal is to predict the category of a given text sample.
```

## Requirements
```
Python 3.x
Keras
TensorFlow
scikit-learn
pandas
numpy
matplotlib
```
## Installation
1. Clone the repository:
```bash
$ git clone <repository_url>
```

2. Install the required dependencies:
```bash
$ pip install -r requirements.txt
```

## Usage
```
1. Prepare the data:

- Place the dataset file All_data_A01.csv in the same directory as the code.
- Update the read_file() function to specify the correct path to the dataset file if needed.
- Implement the dataframe() function to preprocess the dataset and convert the categories to numerical values.
- Update the category mappings in df_num and df_cateEng according to your dataset.
- Save the preprocessed data using pickle in the token() function.

2. Train the model:

- Run the code to train the text classification model using GRU.
- The code performs binary classification for each pair of categories in the dataset.
- The trained models and logs will be saved in the Models_A01 directory.

3. Evaluate the model:

- The code evaluates the trained models on the validation and test datasets.
- The evaluation results are displayed and saved in the log file.
```

## Credits
```
- Author: Shu-Hao Wu
- Email: shwu2@illinois.edu
```
