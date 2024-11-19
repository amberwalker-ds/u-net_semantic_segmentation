# Building Footprint Segmentation with U-Net

This project implements a U-Net convolutional neural network to identify and map building footprints from aerial images. The model was trained on high-resolution aerial imagery and achieves a **94.7% accuracy** and a **76.7% Dice score**, making it suitable for tasks like urban planning and disaster management.

If you would like to a clear and easy-to-understand tutorial, follow along with my Medium article: https://medium.com/ai-advances/how-i-used-a-u-net-to-map-building-footprints-from-the-sky-bf6d184c41d8

## Data Sources
The dataset used in this project contains 3,347 images (256×256×3) of Massachusetts, sourced from OpenStreetMap building footprints. Each image is paired with a binary mask indicating the location of buildings. You can find the raw dataset [here](https://www.cs.toronto.edu/~vmnih/data/).

---

## Installation Instructions
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/amberwalker-ds/u-net_semantic_segmentation
    cd your-repo-name
    ```

2. **Install Dependencies**:
    Ensure you have Python 3.8+ and the required packages installed:
    ```bash
    pip install -r requirements.txt
    ```

3. **Hardware Requirements**:
    - **1 GPU is required** to train and evaluate the U-Net model efficiently.
    - If no GPU is available, use a cloud-based solution like Google Colab or AWS.

---

## Project Structure
.
project/
├── data/
│   ├── test/
│   │   ├── images/         # Test images for evaluation
│   │   ├── labels/         # Corresponding labels (ground truth masks) for test images
│   ├── training/
│   │   ├── images/         # Training images used for model training
│   │   ├── labels/         # Corresponding labels (ground truth masks) for training images
│   ├── validation/
│   │   ├── images/         # Validation images for model tuning
│   │   ├── labels/         # Corresponding labels (ground truth masks) for validation images
│   ├── utilities/
│   │   └── tiles.gpkg      # Geopackage file with additional utility data
├── scripts/                # Python scripts for preprocessing, training, and evaluation
│    └── segmentation_utilities.py  # Helper functions for segmentation tasks
├── notebooks/              # Jupyter notebooks for exploration and prototyping
│   ├── segmentation_practice_amber_walker.ipynb #walkthrough EDA & Model Training
├── README.md               # Project overview and instructions
├── requirements.txt        # List of Python dependencies
└── LICENSE                 # License for the project


---

## Results and Analysis
- **Metrics**:
    - **Accuracy**: 94.7%
    - **Precision**: 75.2%
    - **Recall**: 78.9%
    - **Dice Score**: 76.7%

---

## Limitations and Future Work
### Limitations
1. The model struggles with small buildings or areas with heavy shadows.
2. Limited generalization to aerial images from different regions due to dataset-specific biases.
3. Requires a **GPU** for efficient training and prediction.

### Future Work
1. **Data Augmentation**:
    Introduce more variations (e.g., rotations, brightness adjustments) to improve the model's robustness.
2. **Improved Post-Processing**:
    Refine the predicted masks using morphological operations to reduce false positives and false negatives.
3. **Deployment**:
    Develop a web-based tool with an interactive slider to allow users to upload aerial images and view segmentation results.
---

If you have any questions or feedback, feel free to open an issue or reach out!