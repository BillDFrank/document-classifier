# Document Classifier

This project is a Python-based system for document classification, clustering, and analysis. It provides tools for processing text data, training classifiers, performing clustering, detecting outliers, computing similarities, and generating visualizations. The project is modular and designed for easy extension, making it suitable for tasks like document categorization, search, and recommendation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

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

3. **Set Up a Virtual Environment (recommended)**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main entry point is scripts/run_pipeline.py, which orchestrates the document classification and analysis pipeline.

1. **Prepare Data**:

- Place raw datasets in data/raw/.
- Processed data will be saved in data/processed/.

2. **Run the Pipeline**:
   ```bash
   python scripts/run_pipeline.py
   ```

Outputs (e.g., plots, models) are saved in outputs/ and models/.

## Dependencies

See requirements.txt for the full list of dependencies. Install them using:

```bash
pip install -r requirements.txt
```

## Project Structure

````text
document-classifier/
├── data/
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── src/
│   ├── classification/
│   │   └── classifier.py
│   ├── clustering/
│   │   ├── clusterer.py
│   │   ├── cluster_mover.py
│   │   └── cluster_splitter.py
├── data/
│   │   ├── data_source.py
│   │   └── data_processor.py
│   ├── search/
│   │   ├── search.py
│   │   ├── suggestion.py
│   │   └── auto_suggestion.py
│   ├── analysis/
│   │   ├── outliers.py
│   │   └── similarities.py
│   ├── visualization/
│   │   └── plotter.py
│   ├── utilities/
│   │   └── helpers.py
│   └── init.py
├── notebooks/
│   └── tests.ipynb
├── scripts/
│   └── run_pipeline.py
├── models/
├── outputs/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```" >> README.md
````
