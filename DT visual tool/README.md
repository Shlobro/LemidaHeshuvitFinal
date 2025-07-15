# Decision Tree Visualization Tool - Demo

## Quick Start Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python main.py
```

### 3. Using the Tool

#### Loading Data:
- **Sample Datasets**: Choose from Iris, Wine, or Breast Cancer datasets
- **Custom CSV**: Load your own CSV file (last column should be the target variable)

#### Training a Model:
1. Load your dataset
2. Adjust model parameters in the left panel:
   - Max Depth: Controls tree complexity
   - Min Samples Split: Minimum samples required to split a node
   - Min Samples Leaf: Minimum samples required at leaf node
   - Criterion: 'gini' or 'entropy' for splitting
3. Click "Train Decision Tree"

#### Exploring Results:
- **Data Tab**: View dataset information and sample data
- **Decision Tree Tab**: Interactive tree visualization
- **Results Tab**: Performance metrics and confusion matrix

#### Making Predictions:
1. Enter values for each feature in the prediction panel
2. Click "Predict" to see the classification result and probabilities

### Features:
- ✅ Complete GUI with PySide6
- ✅ Multiple dataset support
- ✅ Interactive decision tree visualization
- ✅ Real-time prediction interface
- ✅ Performance metrics and confusion matrix
- ✅ Adjustable model parameters
- ✅ Data exploration tools

### Example Usage:
1. Start with the Iris dataset for a quick demo
2. Try different max_depth values (3, 5, 10) to see how tree complexity affects visualization
3. Use the prediction panel to classify new flower measurements
4. Compare 'gini' vs 'entropy' criteria in the results tab

This tool provides a complete interface for understanding and visualizing decision tree classifiers!
