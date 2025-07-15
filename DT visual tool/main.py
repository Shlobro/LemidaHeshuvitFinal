import sys
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QFileDialog, QLabel, QTextEdit,
                               QSplitter, QGroupBox, QComboBox, QSpinBox, QTabWidget,
                               QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit,
                               QFormLayout, QScrollArea)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QFont, QColor
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

class DecisionTreeVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decision Tree Visualization Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Data attributes
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.class_names = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # Right panel - Visualization and results
        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)

        # Set initial splitter sizes
        splitter.setSizes([400, 1000])

    def create_control_panel(self):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Data loading section
        data_group = QGroupBox("Data Loading")
        data_layout = QVBoxLayout(data_group)

        # Sample datasets
        self.sample_combo = QComboBox()
        self.sample_combo.addItems(["Load Custom CSV", "Iris Dataset", "Wine Dataset", "Breast Cancer Dataset"])
        data_layout.addWidget(QLabel("Sample Datasets:"))
        data_layout.addWidget(self.sample_combo)

        load_sample_btn = QPushButton("Load Sample Dataset")
        load_sample_btn.clicked.connect(self.load_sample_dataset)
        data_layout.addWidget(load_sample_btn)

        # Custom file loading
        load_file_btn = QPushButton("Load CSV File")
        load_file_btn.clicked.connect(self.load_csv_file)
        data_layout.addWidget(load_file_btn)

        control_layout.addWidget(data_group)

        # Model parameters section
        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout(params_group)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 20)
        self.max_depth_spin.setValue(5)
        params_layout.addRow("Max Depth:", self.max_depth_spin)

        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 20)
        self.min_samples_split_spin.setValue(2)
        params_layout.addRow("Min Samples Split:", self.min_samples_split_spin)

        self.min_samples_leaf_spin = QSpinBox()
        self.min_samples_leaf_spin.setRange(1, 10)
        self.min_samples_leaf_spin.setValue(1)
        params_layout.addRow("Min Samples Leaf:", self.min_samples_leaf_spin)

        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        params_layout.addRow("Criterion:", self.criterion_combo)

        control_layout.addWidget(params_group)

        # Training section
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout(train_group)

        train_btn = QPushButton("Train Decision Tree")
        train_btn.clicked.connect(self.train_model)
        train_layout.addWidget(train_btn)

        control_layout.addWidget(train_group)

        # Prediction section
        pred_group = QGroupBox("Single Prediction")
        pred_layout = QVBoxLayout(pred_group)

        self.prediction_inputs = {}
        self.prediction_form = QWidget()
        self.prediction_form_layout = QFormLayout(self.prediction_form)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.prediction_form)
        scroll_area.setWidgetResizable(True)
        pred_layout.addWidget(scroll_area)

        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.make_prediction)
        pred_layout.addWidget(predict_btn)

        self.prediction_result = QLabel("Prediction will appear here")
        pred_layout.addWidget(self.prediction_result)

        control_layout.addWidget(pred_group)

        control_layout.addStretch()

        return control_widget

    def create_visualization_panel(self):
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)

        # Create tab widget for different views
        self.tab_widget = QTabWidget()
        viz_layout.addWidget(self.tab_widget)

        # Data view tab
        self.create_data_tab()

        # Tree visualization tab
        self.create_tree_tab()

        # Results tab
        self.create_results_tab()

        return viz_widget

    def create_data_tab(self):
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        # Data info
        self.data_info_label = QLabel("No data loaded")
        data_layout.addWidget(self.data_info_label)

        # Data table
        self.data_table = QTableWidget()
        data_layout.addWidget(self.data_table)

        self.tab_widget.addTab(data_widget, "Data")

    def create_tree_tab(self):
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)

        # Tree visualization canvas
        self.tree_figure = Figure(figsize=(12, 8))
        self.tree_canvas = FigureCanvas(self.tree_figure)
        tree_layout.addWidget(self.tree_canvas)

        self.tab_widget.addTab(tree_widget, "Decision Tree")

    def create_results_tab(self):
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Results text
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)

        # Confusion matrix canvas
        self.cm_figure = Figure(figsize=(6, 5))
        self.cm_canvas = FigureCanvas(self.cm_figure)
        results_layout.addWidget(self.cm_canvas)

        self.tab_widget.addTab(results_widget, "Results")

    def load_sample_dataset(self):
        dataset_name = self.sample_combo.currentText()

        if dataset_name == "Load Custom CSV":
            return

        try:
            if dataset_name == "Iris Dataset":
                data = load_iris()
                self.X = pd.DataFrame(data.data, columns=data.feature_names)
                self.y = pd.Series(data.target)
                self.feature_names = data.feature_names
                self.class_names = data.target_names

            elif dataset_name == "Wine Dataset":
                data = load_wine()
                self.X = pd.DataFrame(data.data, columns=data.feature_names)
                self.y = pd.Series(data.target)
                self.feature_names = data.feature_names
                self.class_names = data.target_names

            elif dataset_name == "Breast Cancer Dataset":
                data = load_breast_cancer()
                self.X = pd.DataFrame(data.data, columns=data.feature_names)
                self.y = pd.Series(data.target)
                self.feature_names = data.feature_names
                self.class_names = data.target_names

            self.update_data_display()
            self.setup_prediction_inputs()
            QMessageBox.information(self, "Success", f"{dataset_name} loaded successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset: {str(e)}")

    def load_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv)")

        if file_path:
            try:
                self.data = pd.read_csv(file_path)

                # Assume last column is target
                self.X = self.data.iloc[:, :-1]
                self.y = self.data.iloc[:, -1]
                self.feature_names = self.X.columns.tolist()
                self.class_names = np.unique(self.y).astype(str)

                self.update_data_display()
                self.setup_prediction_inputs()
                QMessageBox.information(self, "Success", "CSV file loaded successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV file: {str(e)}")

    def update_data_display(self):
        if self.X is not None and self.y is not None:
            # Update data info
            info_text = f"Dataset shape: {self.X.shape}\n"
            info_text += f"Features: {len(self.feature_names)}\n"
            info_text += f"Classes: {len(self.class_names)} ({', '.join(map(str, self.class_names))})\n"
            info_text += f"Total samples: {len(self.X)}"
            self.data_info_label.setText(info_text)

            # Update data table
            combined_data = pd.concat([self.X, self.y], axis=1)
            self.data_table.setRowCount(min(100, len(combined_data)))  # Show first 100 rows
            self.data_table.setColumnCount(len(combined_data.columns))
            self.data_table.setHorizontalHeaderLabels(combined_data.columns.astype(str))

            for i in range(min(100, len(combined_data))):
                for j in range(len(combined_data.columns)):
                    item = QTableWidgetItem(str(combined_data.iloc[i, j]))
                    self.data_table.setItem(i, j, item)

    def setup_prediction_inputs(self):
        # Clear existing inputs
        for i in reversed(range(self.prediction_form_layout.count())):
            self.prediction_form_layout.itemAt(i).widget().setParent(None)

        self.prediction_inputs = {}

        if self.feature_names:
            for feature in self.feature_names:
                line_edit = QLineEdit()
                line_edit.setPlaceholderText("Enter value...")
                self.prediction_inputs[feature] = line_edit
                self.prediction_form_layout.addRow(f"{feature}:", line_edit)

    def train_model(self):
        if self.X is None or self.y is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        try:
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )

            # Create and train model
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth_spin.value(),
                min_samples_split=self.min_samples_split_spin.value(),
                min_samples_leaf=self.min_samples_leaf_spin.value(),
                criterion=self.criterion_combo.currentText(),
                random_state=42
            )

            self.model.fit(self.X_train, self.y_train)

            # Update visualizations
            self.visualize_tree()
            self.show_results()

            QMessageBox.information(self, "Success", "Decision tree trained successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train model: {str(e)}")

    def visualize_tree(self):
        if self.model is None:
            return

        try:
            from sklearn.tree import plot_tree

            self.tree_figure.clear()
            ax = self.tree_figure.add_subplot(111)

            plot_tree(self.model,
                     feature_names=self.feature_names,
                     class_names=[str(name) for name in self.class_names],
                     filled=True,
                     rounded=True,
                     fontsize=8,
                     ax=ax)

            ax.set_title("Decision Tree Visualization", fontsize=14, fontweight='bold')
            self.tree_figure.tight_layout()
            self.tree_canvas.draw()

        except Exception as e:
            print(f"Error visualizing tree: {str(e)}")

    def show_results(self):
        if self.model is None:
            return

        try:
            # Predictions
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)

            # Classification report
            report = classification_report(self.y_test, y_pred,
                                         target_names=[str(name) for name in self.class_names])

            # Display results
            results_text = f"Model Performance\n{'='*50}\n\n"
            results_text += f"Accuracy: {accuracy:.4f}\n\n"
            results_text += f"Classification Report:\n{report}\n\n"
            results_text += f"Model Parameters:\n"
            results_text += f"- Max Depth: {self.model.max_depth}\n"
            results_text += f"- Min Samples Split: {self.model.min_samples_split}\n"
            results_text += f"- Min Samples Leaf: {self.model.min_samples_leaf}\n"
            results_text += f"- Criterion: {self.model.criterion}\n"

            self.results_text.setText(results_text)

            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            self.plot_confusion_matrix(cm)

        except Exception as e:
            print(f"Error showing results: {str(e)}")

    def plot_confusion_matrix(self, cm):
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[str(name) for name in self.class_names],
                   yticklabels=[str(name) for name in self.class_names],
                   ax=ax)

        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        self.cm_figure.tight_layout()
        self.cm_canvas.draw()

    def make_prediction(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please train a model first!")
            return

        try:
            # Get input values
            input_values = []
            for feature in self.feature_names:
                value_text = self.prediction_inputs[feature].text()
                if not value_text:
                    QMessageBox.warning(self, "Warning", f"Please enter a value for {feature}")
                    return
                input_values.append(float(value_text))

            # Make prediction
            prediction = self.model.predict([input_values])[0]
            probabilities = self.model.predict_proba([input_values])[0]

            # Display result
            result_text = f"Predicted Class: {self.class_names[prediction]}\n\n"
            result_text += "Class Probabilities:\n"
            for i, prob in enumerate(probabilities):
                result_text += f"{self.class_names[i]}: {prob:.4f}\n"

            self.prediction_result.setText(result_text)

        except ValueError as e:
            QMessageBox.warning(self, "Warning", "Please enter valid numeric values for all features!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = DecisionTreeVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
