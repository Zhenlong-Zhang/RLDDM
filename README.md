# Behavior Modeling Project

This project contains both **MATLAB** and **Python** scripts/notebooks for modeling behavioral data and analyzing reaction times and learning parameters. All files are designed to run **without needing to manually set paths**—as long as they are placed in the **same working directory**, everything should work smoothly.

## File Structure

- MATLAB scripts: for data cleaning and model fitting.

- Python notebooks: for model comparison, parameter correlation, parameter analysis, and graphing.

- requirements.txt: lists Python dependencies (see below for setup instructions).

## Repository

### Clone this repository to get started:

```bash
git clone https://github.com/Zhenlong-Zhang/RLDDM.git
cd RLDDM
```
### (Optional) Create and activate a virtual environment for python useage

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

## Python Setup
You can install the required Python libraries in three ways:

### Option 1: Using requirements.txt

```bash
pip install -r requirements.txt
```

### Option 2: Install directly from within each notebook
Each Jupyter notebook includes a **!pip install** cell at the top.

### Option 3: Using environment.yml with conda
```bash
conda env create -f environment.yml
conda activate rlddm-env
```
This will install all required packages and Python version as specified in the YAML file.

## Matlab Setup

- Recommended version: MATLAB R2021a or later
  
- Required toolbox: Optimization Toolbox (for fmincon)

## Matlab Usage
- Place all data files and .m scripts in the same working folder.

- No need to configure any paths.

- Data structure: (TBD – to be documented)
