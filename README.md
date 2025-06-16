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
You can install the required Python libraries in two ways:

### Option 1: Using requirements.txt (recommended)

```bash
pip install -r requirements.txt
```

### Option 2: Install directly from within each notebook
Each Jupyter notebook includes a **!pip install** cell at the top\.

## Matlab Setup

- Recommended version: MATLAB R2021a or later
  
- Required toolbox: Optimization Toolbox (for fmincon)

### Matlab Usage
- Place all data files and .m scripts in the same working folder.

- No need to configure any paths.

- Data structure: (TBD – to be documented)
