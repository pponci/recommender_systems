# Recommender Systems Project

This project implements a recommendation system using collaborative filtering, matrix factorization (ALS), and a simple web interface built with Streamlit.

It is structured around the following key components:

- ALS-based collaborative filtering using the `implicit` library  
- Data processing and evaluation scripts
- A web app using **Streamlit**

---

## Environment Setup (Conda)

We use a Conda environment to manage dependencies.  
To get started, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### 1. Clone the repository

```bash
git clone https://github.com/pponci/recommender_systems.git
cd recommender_systems
```

### 2. Create and activate the environment

```bash
conda env create -f environment.yaml
conda activate recommender_env
```

This will install all required dependencies including implicit, pandas, scikit-learn, and streamlit.

### 3. Run the app

```bash
streamlit run app/streamlitapp.py
```
