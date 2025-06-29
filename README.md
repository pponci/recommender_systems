# Recommender Systems Project

## Description

This project implements a recommendation system using collaborative filtering, matrix factorization (ALS), and a simple web interface built with Streamlit. The system leverages ALS-based collaborative filtering using the `implicit` library along with data processing and evaluation scripts.

## Project Structure

```
RECOMMENDER_SYSTEMS/
├── app/                    # Streamlit application
│   └── streamlitapp.py    # Main user interface
├── data/                   # Project datasets
├── models/                 # Trained models
├── src/                    # Main source code
├── environment.yml         # Conda dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker orchestration
├── .gitignore            # Git ignored files
├── .dockerignore         # Docker ignored files
└── README.md             # Project documentation
```

## Key Components

- **ALS-based collaborative filtering** using the `implicit` library  
- **Data processing and evaluation scripts**
- **Web app** using Streamlit for interactive recommendations
- **Docker containerization** for easy deployment

## Installation and Usage

### Option 1: Local Installation (Conda)

We use a Conda environment to manage dependencies. Make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

#### 1. Clone the repository

```bash
git clone https://github.com/pponci/recommender_systems.git
cd recommender_systems
```

#### 2. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate recommender_env
```

**Alternative (faster):** If the installation is too slow with conda, you can use mamba:

```bash
conda install -n base -c conda-forge mamba
mamba env create -f environment.yml
conda activate recommender_env
```

This will install all required dependencies including implicit, pandas, scikit-learn, and streamlit.

#### 3. Run the app

```bash
streamlit run app/streamlitapp.py
```

### Option 2: With Docker (Recommended for Production)

Docker provides a consistent environment and simplifies deployment.

#### Prerequisites
- Docker installed on your machine
- Docker Compose (optional)

#### Quick Start
```bash
# Clone the repository
git clone https://github.com/pponci/recommender_systems.git
cd recommender_systems

# Build and run the application
docker build -t recommender_app .
docker run -p 8501:8501 recommender_app
```

#### With Docker Compose
```bash
# Launch the application
docker-compose up --build

# Run in background
docker-compose up -d --build
```

#### Access the Application
Once launched, the application is accessible at: **http://localhost:8501**

## Technologies Used

- **Python 3.10** - Main programming language
- **Streamlit** - Web user interface
- **implicit** - ALS-based collaborative filtering algorithms
- **scikit-learn** - Machine learning algorithms and evaluation
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scipy** - Scientific computing
- **Docker** - Containerization
- **Conda/Mamba** - Environment management

## Features

- **ALS Matrix Factorization** - Collaborative filtering recommendations
- **Interactive Web Interface** - User-friendly Streamlit application
- **Data Processing Pipeline** - Automated data cleaning and preprocessing
- **Model Evaluation** - Built-in metrics and validation
- **Docker Support** - Easy deployment and scaling
- **REST API Integration** - Ready for production deployment

## Development

### Code Structure
- `app/streamlitapp.py` - Main Streamlit user interface
- `src/` - Algorithm source code and data processing
- `models/` - Trained ALS models and saved artifacts
- `data/` - Datasets and processed data files

### Run in Development Mode
```bash
# With automatic reload
streamlit run app/streamlitapp.py --server.runOnSave=true
```

### Performance Optimization
For faster environment setup, consider using **mamba** instead of conda:
```bash
conda install -n base -c conda-forge mamba
mamba env create -f environment.yml
```

## Docker

### Build the Image
```bash
docker build -t recommender_app:latest .
```

### Health Check
The application includes an automatic health check accessible via:
`http://localhost:8501/_stcore/health`

## Deployment Notes

- Application listens on port 8501
- Production environment configuration included
- Health checks configured for monitoring
- Optimized for low resource consumption

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Authors

- Piercalo PONCI
- Timothée ROSSILLON
- Muhammad Adam Hakeem BIN MOHAMAD NAZRI
- Sema Nur OZEN
- Lou DE GAETANO NERINO DE VISCONTI

## Support

If you encounter issues:

### Local Installation Issues
1. Ensure Conda/Miniconda is properly installed
2. Try using mamba for faster dependency resolution
3. Check that all dependencies in `environment.yml` are available

### Docker Issues
1. Verify that Docker is properly installed and running
2. Ensure port 8501 is available
3. Check Docker logs: `docker logs <container_id>`

### Application Issues
1. Verify that the Streamlit app starts without errors
2. Check browser console for JavaScript errors
3. Ensure all required data files are present

For additional help, open an issue on GitHub: https://github.com/pponci/recommender_systems/issues