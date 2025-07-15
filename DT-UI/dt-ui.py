import sys
import math
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                               QWidget, QPushButton, QTextEdit, QLabel, QComboBox,
                               QTableWidget, QTableWidgetItem, QSplitter, QTabWidget,
                               QScrollArea, QGroupBox, QSlider, QSpinBox, QProgressBar)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None,
                 samples=None, gini=None, entropy=None, class_counts=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value  # For leaf nodes
        self.left = left
        self.right = right
        self.samples = samples
        self.gini = gini
        self.entropy = entropy
        self.class_counts = class_counts


class DecisionTreeVisualizer:
    def __init__(self):
        self.tree = None
        self.max_depth = 3
        self.min_samples_split = 2
        self.criterion = 'gini'

    def calculate_gini(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0

        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - sum(p ** 2 for p in probabilities)
        return gini

    def calculate_entropy(self, y):
        """Calculate entropy"""
        if len(y) == 0:
            return 0

        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy

    def calculate_information_gain(self, X, y, feature_idx, threshold):
        """Calculate information gain for a split"""
        if self.criterion == 'gini':
            parent_impurity = self.calculate_gini(y)
        else:
            parent_impurity = self.calculate_entropy(y)

        # Split the data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if sum(left_mask) == 0 or sum(right_mask) == 0:
            return 0

        left_y = y[left_mask]
        right_y = y[right_mask]

        # Calculate weighted impurity
        n_samples = len(y)
        left_weight = len(left_y) / n_samples
        right_weight = len(right_y) / n_samples

        if self.criterion == 'gini':
            left_impurity = self.calculate_gini(left_y)
            right_impurity = self.calculate_gini(right_y)
        else:
            left_impurity = self.calculate_entropy(left_y)
            right_impurity = self.calculate_entropy(right_y)

        weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
        information_gain = parent_impurity - weighted_impurity

        return information_gain

    def find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                gain = self.calculate_information_gain(X, y, feature_idx, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, depth=0):
        """Build the decision tree recursively"""
        n_samples = len(y)

        # Calculate impurity metrics
        gini = self.calculate_gini(y)
        entropy = self.calculate_entropy(y)

        # Get class counts
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))

        # Base cases
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                len(np.unique(y)) == 1):
            # Return leaf node
            most_common_class = classes[np.argmax(counts)]
            return DecisionTreeNode(
                value=most_common_class,
                samples=n_samples,
                gini=gini,
                entropy=entropy,
                class_counts=class_counts
            )

        # Find best split
        best_feature, best_threshold, best_gain = self.find_best_split(X, y)

        if best_feature is None or best_gain <= 0:
            # Return leaf node
            most_common_class = classes[np.argmax(counts)]
            return DecisionTreeNode(
                value=most_common_class,
                samples=n_samples,
                gini=gini,
                entropy=entropy,
                class_counts=class_counts
            )

        # Create internal node
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            samples=n_samples,
            gini=gini,
            entropy=entropy,
            class_counts=class_counts
        )

    def fit(self, X, y):
        """Fit the decision tree to the data"""
        self.tree = self.build_tree(X, y)
        return self.tree


class TreeCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.tree = None
        self.feature_names = None
        self.class_names = None
        self.setMinimumSize(800, 600)

    def set_tree(self, tree, feature_names=None, class_names=None):
        self.tree = tree
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(10)]
        self.class_names = class_names or ["Class_0", "Class_1"]
        self.update()

    def paintEvent(self, event):
        if not self.tree:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate tree dimensions
        self.node_width = 150
        self.node_height = 80
        self.level_height = 120

        # Draw tree
        self.draw_tree(painter, self.tree, self.width() // 2, 50, self.width() // 4)

    def draw_tree(self, painter, node, x, y, x_offset):
        if not node:
            return

        # Draw node
        self.draw_node(painter, node, x, y)

        # Draw children
        if node.left:
            # Draw left branch
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawLine(x - 20, y + self.node_height,
                             x - x_offset, y + self.level_height)

            # Draw left child
            self.draw_tree(painter, node.left, x - x_offset,
                           y + self.level_height, x_offset // 2)

        if node.right:
            # Draw right branch
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            painter.drawLine(x + 20, y + self.node_height,
                             x + x_offset, y + self.level_height)

            # Draw right child
            self.draw_tree(painter, node.right, x + x_offset,
                           y + self.level_height, x_offset // 2)

    def draw_node(self, painter, node, x, y):
        # Node background
        if node.value is not None:  # Leaf node
            painter.setBrush(QBrush(QColor(144, 238, 144)))  # Light green
        else:  # Internal node
            painter.setBrush(QBrush(QColor(173, 216, 230)))  # Light blue

        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawRoundedRect(x - self.node_width // 2, y,
                                self.node_width, self.node_height, 10, 10)

        # Node text
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.setFont(QFont("Arial", 9))

        text_x = x - self.node_width // 2 + 5
        text_y = y + 15

        if node.value is not None:  # Leaf node
            painter.drawText(text_x, text_y, f"Class: {self.class_names[node.value]}")
        else:  # Internal node
            feature_name = self.feature_names[node.feature]
            painter.drawText(text_x, text_y, f"{feature_name} <= {node.threshold:.2f}")

        # Draw metrics
        painter.drawText(text_x, text_y + 15, f"Samples: {node.samples}")
        painter.drawText(text_x, text_y + 30, f"Gini: {node.gini:.3f}")
        painter.drawText(text_x, text_y + 45, f"Entropy: {node.entropy:.3f}")


class StepByStepWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Step information
        self.step_label = QLabel("Step-by-step Decision Tree Construction")
        self.step_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(self.step_label)

        # Text area for detailed calculations
        self.calculation_text = QTextEdit()
        self.calculation_text.setReadOnly(True)
        self.calculation_text.setMaximumHeight(200)
        self.layout.addWidget(self.calculation_text)

        # Progress bar
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)

        # Controls
        controls_layout = QHBoxLayout()
        self.next_button = QPushButton("Next Step")
        self.prev_button = QPushButton("Previous Step")
        self.reset_button = QPushButton("Reset")

        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.reset_button)

        self.layout.addLayout(controls_layout)

        # Current step tracking
        self.current_step = 0
        self.steps = []

    def set_steps(self, steps):
        self.steps = steps
        self.current_step = 0
        self.progress.setMaximum(len(steps) - 1)
        self.update_display()

    def update_display(self):
        if self.steps and 0 <= self.current_step < len(self.steps):
            step = self.steps[self.current_step]
            self.calculation_text.setHtml(step)
            self.progress.setValue(self.current_step)

            # Update button states
            self.prev_button.setEnabled(self.current_step > 0)
            self.next_button.setEnabled(self.current_step < len(self.steps) - 1)


class DatasetWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Select Dataset:"))

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Simple Binary", "Iris (simplified)", "Weather", "Custom"
        ])
        dataset_layout.addWidget(self.dataset_combo)

        self.layout.addLayout(dataset_layout)

        # Data table
        self.data_table = QTableWidget()
        self.layout.addWidget(self.data_table)

        # Initialize with simple dataset
        self.load_dataset()

    def load_dataset(self):
        dataset_name = self.dataset_combo.currentText()

        if dataset_name == "Simple Binary":
            self.data = {
                'Feature_0': [1, 2, 3, 4, 5, 6, 7, 8],
                'Feature_1': [2, 3, 1, 4, 5, 2, 6, 7],
                'Class': [0, 0, 0, 1, 1, 0, 1, 1]
            }
        elif dataset_name == "Iris (simplified)":
            self.data = {
                'Sepal_Length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0],
                'Sepal_Width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4],
                'Class': [0, 0, 0, 0, 1, 1, 1, 1]
            }
        elif dataset_name == "Weather":
            self.data = {
                'Temperature': [85, 80, 83, 70, 68, 65, 64, 72],
                'Humidity': [85, 90, 78, 96, 80, 70, 65, 95],
                'Play': [0, 0, 1, 1, 1, 1, 1, 0]
            }

        self.update_table()

    def update_table(self):
        df = pd.DataFrame(self.data)
        self.data_table.setRowCount(len(df))
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                self.data_table.setItem(i, j, QTableWidgetItem(str(value)))

    def get_data(self):
        return self.data


class DecisionTreeLearningTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Decision Tree Learning Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize components
        self.tree_visualizer = DecisionTreeVisualizer()
        self.current_tree = None
        self.current_data = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Controls and dataset
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Dataset widget
        self.dataset_widget = DatasetWidget()
        left_layout.addWidget(self.dataset_widget)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout()
        params_group.setLayout(params_layout)

        # Criterion
        criterion_layout = QHBoxLayout()
        criterion_layout.addWidget(QLabel("Criterion:"))
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy"])
        criterion_layout.addWidget(self.criterion_combo)
        params_layout.addLayout(criterion_layout)

        # Max depth
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Max Depth:"))
        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setRange(1, 10)
        self.depth_spinbox.setValue(3)
        depth_layout.addWidget(self.depth_spinbox)
        params_layout.addLayout(depth_layout)

        # Min samples split
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(QLabel("Min Samples Split:"))
        self.samples_spinbox = QSpinBox()
        self.samples_spinbox.setRange(2, 10)
        self.samples_spinbox.setValue(2)
        samples_layout.addWidget(self.samples_spinbox)
        params_layout.addLayout(samples_layout)

        left_layout.addWidget(params_group)

        # Build tree button
        self.build_button = QPushButton("Build Tree")
        self.build_button.clicked.connect(self.build_tree)
        left_layout.addWidget(self.build_button)

        # Step-by-step widget
        self.step_widget = StepByStepWidget()
        self.step_widget.next_button.clicked.connect(self.next_step)
        self.step_widget.prev_button.clicked.connect(self.prev_step)
        self.step_widget.reset_button.clicked.connect(self.reset_steps)
        left_layout.addWidget(self.step_widget)

        main_layout.addWidget(left_panel)

        # Right panel - Visualizations
        right_panel = QTabWidget()

        # Tree visualization
        self.tree_canvas = TreeCanvas()
        right_panel.addTab(self.tree_canvas, "Tree Structure")

        # Data visualization
        self.data_canvas = self.create_data_canvas()
        right_panel.addTab(self.data_canvas, "Data Visualization")

        main_layout.addWidget(right_panel)

        # Connect signals
        self.dataset_widget.dataset_combo.currentTextChanged.connect(
            self.dataset_widget.load_dataset
        )

    def create_data_canvas(self):
        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout()
        canvas_widget.setLayout(canvas_layout)

        # Matplotlib figure
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        canvas_layout.addWidget(self.canvas)

        return canvas_widget

    def build_tree(self):
        # Get data
        data = self.dataset_widget.get_data()
        df = pd.DataFrame(data)

        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Class', 'Play']]
        target_col = 'Class' if 'Class' in df.columns else 'Play'

        X = df[feature_cols].values
        y = df[target_col].values

        # Set parameters
        self.tree_visualizer.criterion = self.criterion_combo.currentText()
        self.tree_visualizer.max_depth = self.depth_spinbox.value()
        self.tree_visualizer.min_samples_split = self.samples_spinbox.value()

        # Build tree
        self.current_tree = self.tree_visualizer.fit(X, y)
        self.current_data = (X, y, feature_cols, target_col)

        # Update visualizations
        self.update_tree_visualization()
        self.update_data_visualization()
        self.generate_steps()

    def update_tree_visualization(self):
        if self.current_tree and self.current_data:
            X, y, feature_cols, target_col = self.current_data
            unique_classes = np.unique(y)
            class_names = [f"Class_{c}" for c in unique_classes]

            self.tree_canvas.set_tree(self.current_tree, feature_cols, class_names)

    def update_data_visualization(self):
        if not self.current_data:
            return

        X, y, feature_cols, target_col = self.current_data

        self.fig.clear()

        if X.shape[1] >= 2:
            ax = self.fig.add_subplot(111)

            # Create scatter plot
            unique_classes = np.unique(y)
            colors = ['red', 'blue', 'green', 'orange']

            for i, class_val in enumerate(unique_classes):
                mask = y == class_val
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=colors[i % len(colors)],
                           label=f'Class {class_val}', alpha=0.7)

            ax.set_xlabel(feature_cols[0])
            ax.set_ylabel(feature_cols[1])
            ax.set_title('Data Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def generate_steps(self):
        if not self.current_tree or not self.current_data:
            return

        X, y, feature_cols, target_col = self.current_data
        steps = []

        # Generate step-by-step explanation
        steps.append(self.generate_initial_step(X, y))
        self.generate_tree_steps(self.current_tree, X, y, feature_cols, steps, depth=0)

        self.step_widget.set_steps(steps)

    def generate_initial_step(self, X, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        # Calculate initial impurity
        gini = self.tree_visualizer.calculate_gini(y)
        entropy = self.tree_visualizer.calculate_entropy(y)

        step_html = f"""
        <h3>Step 1: Initial Dataset Analysis</h3>
        <p><b>Total Samples:</b> {total_samples}</p>
        <p><b>Class Distribution:</b></p>
        <ul>
        """

        for cls, count in zip(unique_classes, counts):
            probability = count / total_samples
            step_html += f"<li>Class {cls}: {count} samples ({probability:.3f})</li>"

        step_html += f"""
        </ul>
        <p><b>Initial Impurity Measures:</b></p>
        <ul>
        <li>Gini Impurity: {gini:.3f}</li>
        <li>Entropy: {entropy:.3f}</li>
        </ul>

        <p><b>Gini Calculation:</b> 1 - Σ(p_i²)</p>
        <p><b>Entropy Calculation:</b> -Σ(p_i × log₂(p_i))</p>
        """

        return step_html

    def generate_tree_steps(self, node, X, y, feature_names, steps, depth=0):
        if node.value is not None:  # Leaf node
            return

        # Generate step for this split
        feature_name = feature_names[node.feature]
        threshold = node.threshold

        # Calculate information gain details
        left_mask = X[:, node.feature] <= threshold
        right_mask = ~left_mask

        left_y = y[left_mask]
        right_y = y[right_mask]

        if self.tree_visualizer.criterion == 'gini':
            parent_impurity = self.tree_visualizer.calculate_gini(y)
            left_impurity = self.tree_visualizer.calculate_gini(left_y)
            right_impurity = self.tree_visualizer.calculate_gini(right_y)
        else:
            parent_impurity = self.tree_visualizer.calculate_entropy(y)
            left_impurity = self.tree_visualizer.calculate_entropy(left_y)
            right_impurity = self.tree_visualizer.calculate_entropy(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
        information_gain = parent_impurity - weighted_impurity

        step_html = f"""
        <h3>Step {len(steps) + 1}: Split on {feature_name} <= {threshold:.2f}</h3>
        <p><b>Parent Node:</b> {len(y)} samples</p>
        <p><b>Parent {self.tree_visualizer.criterion.capitalize()}:</b> {parent_impurity:.3f}</p>

        <p><b>After Split:</b></p>
        <ul>
        <li>Left child: {len(left_y)} samples ({left_weight:.3f}), {self.tree_visualizer.criterion}: {left_impurity:.3f}</li>
        <li>Right child: {len(right_y)} samples ({right_weight:.3f}), {self.tree_visualizer.criterion}: {right_impurity:.3f}</li>
        </ul>

        <p><b>Weighted {self.tree_visualizer.criterion.capitalize()}:</b> {weighted_impurity:.3f}</p>
        <p><b>Information Gain:</b> {information_gain:.3f}</p>

        <p><i>This was the best split among all possible features and thresholds.</i></p>
        """

        steps.append(step_html)

        # Recursively generate steps for children
        if node.left:
            self.generate_tree_steps(node.left, X[left_mask], left_y, feature_names, steps, depth + 1)
        if node.right:
            self.generate_tree_steps(node.right, X[right_mask], right_y, feature_names, steps, depth + 1)

    def next_step(self):
        if self.step_widget.current_step < len(self.step_widget.steps) - 1:
            self.step_widget.current_step += 1
            self.step_widget.update_display()

    def prev_step(self):
        if self.step_widget.current_step > 0:
            self.step_widget.current_step -= 1
            self.step_widget.update_display()

    def reset_steps(self):
        self.step_widget.current_step = 0
        self.step_widget.update_display()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = DecisionTreeLearningTool()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()