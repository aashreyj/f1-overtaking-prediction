<H1 style="text-align: center;"> CS7.403b: Statistical Methods in AI </H1>
<H3 style="text-align: center;"> Final Project: AI For Formula One </H3>


This machine learning project predicts overtakes in Formula 1 races by analyzing telemetry and event data. Using an XGBoost classifier, we can predict whether an overtake will occur based on various race conditions and car performance metrics.

## Project Overview

Formula 1 overtaking is influenced by numerous factors including car performance, track characteristics, tire conditions, and driver skill. This model aims to quantify these relationships and provide predictive insights that could be valuable for race strategy planning.

## Features

- **Data Preprocessing**: Clean and normalize raw F1 telemetry data
- **Feature Engineering**: Extract and transform relevant features for optimal model performance
- **Model Training**: Train and optimize an XGBoost classifier for overtake prediction
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Prediction System**: Make predictions on new race scenarios

## Model Weights and Dataset
Download the model weights and datasets from:
[Google Drive - F1 Overtake Prediction Files](https://drive.google.com/drive/folders/1Lq3HhjN9XMZBNVjsL7g6xNIYtB4iHR0g?usp=sharing)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
   ```bash
   git clone git@github.com:aashreyj/f1-overtaking-prediction.git
   cd f1-overtake-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model weights and datasets from:
   [Google Drive - F1 Overtake Prediction Files](https://drive.google.com/drive/folders/1Lq3HhjN9XMZBNVjsL7g6xNIYtB4iHR0g?usp=sharing)


## Usage

### Running the Project

#### Exploration and Experimentation
1. Start Jupyter Lab or Notebook:
   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

2. Open `experiments.ipynb` to explore the data and experiment with different approaches.

#### Data Preprocessing
Process the raw data using the preprocessing scripts:
```bash
python preprocessing/process_dataset_overtakes.py
python preprocessing/process_dataset_not_overtakes.py
```

#### Training
Train the model using:
```bash
python train.py
```

#### Evaluation
Evaluate the model's performance:
```bash
python eval.py
```

#### Inference
Make predictions on new data:
```bash
python infer.py
```

### Making Predictions

You can make predictions in this ways:

1. Using the `infer.py` script:
```bash
python infer.py --input your_input_data.csv --output predictions.csv
```

## Dataset Description

The project uses two main processed datasets:
- `processed_data_overtakes.csv`: Contains data points where overtakes occurred
- `processed_data_no_overtakes.csv`: Contains data points where no overtakes occurred


### Data Visualizations

The `resources/` directory contains several visualizations to help understand the data:
- `correlation_matrix.png`: Shows feature correlations
- `overtakes_by_circuit.png`: Displays the distribution of overtakes across different circuits
- `top_circuits_for_overtaking.png`: Highlights the circuits with the most overtaking opportunities
- `ham_all_laps.png` and `ham_lap_9.png`: Visualizations of Lewis Hamilton's racing data
