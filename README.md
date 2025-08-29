# Document Classifier

A comprehensive Python-based system for document classification, text comparison, clustering, and analysis. This project provides multiple Streamlit applications for processing text data, training classifiers, performing clustering, detecting similarities, and generating visualizations.

## ✨ Key Features

- **Text Classification**: Train and use machine learning models for document categorization
- **Text Comparison**: Compare texts using advanced sentence transformers with Excel sheet selection
- **Document Clustering**: Group similar documents using K-means clustering
- **Search & Suggestions**: Intelligent document search with auto-suggestions
- **Data Processing**: Convert and process various data formats
- **Excel Sheet Selection**: Choose specific sheets from Excel files for comparison
- **Interactive Visualizations**: Generate plots and charts for data analysis

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Applications](#applications)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Features](#features)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/BillDFrank/document-classifier.git
   cd document-classifier
   ```

2. **Set Up a Virtual Environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main entry point is `run_pipeline.py`, which provides access to all applications through a unified interface.

1. **Run the Application**:
   ```bash
   streamlit run run_pipeline.py
   ```

2. **Prepare Data**:
   - Place raw datasets in `data/raw/`
   - Processed data will be saved in `data/processed/`
   - Models are saved in `models/`
   - Outputs (plots, results) are saved in `outputs/`

## Applications

### 🔍 Text Comparator (NEW!)
- Compare texts from CSV or Excel files using sentence transformers
- **NEW**: Select specific Excel sheets before comparison
- Advanced similarity scoring with customizable thresholds
- Download comparison results as CSV

### 📊 Document Classifier
- Train machine learning models for text classification
- Support for multiple algorithms (SVM, Random Forest, Neural Networks, etc.)
- Model evaluation with detailed metrics and confusion matrices
- Save and load trained models

### 🎯 Clustering
- Group similar documents using K-means clustering
- Interactive cluster visualization
- Export clustered data

### 🔎 Search & Suggestions
- Intelligent document search functionality
- Auto-suggestion system for improved user experience
- Fast text-based search capabilities

### 📁 Data Processing
- Convert between different data formats
- Process and clean text data
- Handle various file types (CSV, Excel, JSON)

## Project Structure

```
document-classifier/
├── data/
│   ├── raw/              # Raw input data
│   └── processed/        # Processed data
├── src/
│   ├── classification/
│   │   ├── classifier.py              # Main classifier application
│   │   ├── classifier_training.py     # Model training interface
│   │   ├── text_classifier_app.py     # Text classification app
│   │   └── text_comparator_app.py     # Text comparison with Excel sheet selection
│   ├── clustering/
│   │   └── clusterer.py               # Document clustering
│   ├── data/
│   │   ├── data_source.py             # Data source management
│   │   └── data_processor.py          # Data processing utilities
│   ├── search/
│   │   ├── search.py                  # Search functionality
│   │   └── suggestion.py              # Auto-suggestions
│   ├── analysis/
│   │   └── similarities.py            # Similarity analysis
│   ├── utilities/
│   │   ├── converter.py               # Data format conversion
│   │   └── helpers.py                 # Helper functions
│   └── __init__.py
├── models/                            # Saved ML models
├── outputs/                           # Generated outputs and plots
├── notebooks/                         # Jupyter notebooks
├── run_pipeline.py                    # Main application entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── .gitignore
```

## Dependencies

The project uses the following core dependencies:

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **sentence-transformers**: Text similarity and embeddings
- **transformers**: NLP models and tokenizers
- **torch**: Deep learning framework
- **plotly**: Interactive visualizations
- **openpyxl**: Excel file handling
- **joblib**: Model serialization

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Features

### Excel Sheet Selection (Latest Addition)
The text comparator now supports selecting specific sheets from Excel files:
- Upload Excel files with multiple worksheets
- Choose which sheet to use for comparison
- Seamless integration with existing CSV functionality
- Improved user experience for multi-sheet Excel documents

### Text Similarity Comparison
- Uses advanced sentence transformers for accurate text similarity
- Customizable similarity thresholds
- Batch processing of multiple text pairs
- Export results with similarity scores

### Machine Learning Pipeline
- Multiple classification algorithms
- Cross-validation support
- Model performance metrics
- Confusion matrix visualization

### Interactive Interface
- Streamlit-based web interface
- Real-time results and visualizations
- User-friendly file upload system
- Download capabilities for results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.
