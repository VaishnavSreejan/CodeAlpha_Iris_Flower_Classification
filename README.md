# Iris Flower Classification

## Project Overview
This is a professional internship-level Data Science project, designed to build a classification pipeline predicting the species of an Iris flower. Through this project, we demonstrate strong capabilities in Python data manipulation, exploratory data analysis, visual storytelling, and implementing state-of-the-art machine learning models using `scikit-learn`.

## Problem Statement
Biologists often need to categorize species based on observable traits. The problem being solved here is automating this classification process. We want to predict which of the three species an Iris flower belongs to strictly using four numerical measurements of its petals and sepals. 

## Dataset
- **Dataset Source:** Built into Scikit-learn (`sklearn.datasets.load_iris`).
- **Number of Samples:** 150 unique flowers.
- **Four Features:** 
  1. Sepal Length (cm)
  2. Sepal Width (cm)
  3. Petal Length (cm)
  4. Petal Width (cm)
- **Three Target Classes:** 
  - 0: `setosa`
  - 1: `versicolor`
  - 2: `virginica`

## Technologies Used
- **Python 3**
- **Pandas** (Data manipulation)
- **NumPy** (Numerical operations)
- **Matplotlib / Seaborn** (Data visualization)
- **Scikit-learn** (Machine learning algorithms and preprocessing)
- **Jupyter Notebook** (Interactive analysis logging)

## Project Workflow
Data Collection 
↓ 
Data Exploration 
↓ 
Data Cleaning 
↓ 
Visualization 
↓ 
Preprocessing 
↓ 
Train-Test Split 
↓ 
Model Training 
↓ 
Evaluation 
↓ 
Prediction

## Machine Learning Models
We compare four fundamental algorithms:
- **Logistic Regression:** A linear model that computes the probability of a flower belonging to a certain species.
- **K-Nearest Neighbors (KNN):** A distance-based model predicting the species by finding the 'K' most similar flowers in the training data.
- **Decision Tree:** A non-linear tree-based method splitting data into decision nodes.
- **Random Forest:** An ensemble method utilizing multiple Decision Trees to prevent overfitting and improve robustness.

## Evaluation Metrics
When comparing models, we use:
- **Accuracy:** The ratio of correctly predicted flowers over the total tested flowers.
- **Precision:** Of all the flowers the model *predicted* as a particular species, how many were actually that species?
- **Recall:** Of all the *actual* flowers of a particular species, how many did the model find?
- **F1-score:** The harmonic mapping uniting Precision and Recall.
- **Confusion Matrix:** A grid to easily visualize exactly where the model got confused (i.e. predicted Virginica instead of Versicolor).

## Results
The performance of the models on the unseen Test data was exceptionally high. For instance, the **Logistic Regression** and **KNN** models typically achieve >90% precision and accuracy on the test set, proving that petal/sepal geometry is an extraordinarily reliable predictor of Iris species. Please see `outputs/results/model_comparison.csv` and the generated plots for the exact numerical metrics yielded by the current dataset random-split.

## How to Run

1. **Create and Activate Virtual Environment** (Windows PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. **Install Dependencies:**
```powershell
pip install -r requirements.txt
```

3. **Run the Project Pipeline:**
```powershell
python src/main.py
```
*This command executes the entire lifecycle, populates the outputs directory, and demonstrates sample predictions.*

4. **Starting Jupyter Notebook:**
```powershell
jupyter notebook
```

## Project Structure
```text
CodeAlpha_Iris_Flower_Classification/
├── data/
├── notebooks/
│   └── iris_analysis.ipynb
├── outputs/
│   ├── models/
│   ├── plots/
│   └── results/
├── src/
│   ├── data_loader.py
│   ├── main.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── visualization.py
├── .gitignore
├── INTERNSHIP_EXPLANATION.md
├── PROJECT_PRESENTATION.md
├── README.md
└── requirements.txt
```

## Example Prediction
A sample run of the prediction function (`predict_species`) embedded in our program:
- **Input Features (cm):** Sepal length=5.1, Sepal width=3.5, Petal length=1.4, Petal width=0.2
- **Model Output:** `setosa`

## Conclusion
This project successfully verifies that using structural botanical measurements provides highly statistically significant separation between Iris species. Specifically, Iris Setosa is linearly separated from the rest based primarily on its small petal length, whereas Versicolor and Virginica slightly intersect spatially. Scikit-learn’s classifiers effortlessly captured these boundaries yielding near-perfect classifications.

## Future Improvements
- **Trying Additional Algorithms:** Applying Support Vector Machines (SVM).
- **Hyperparameter Tuning:** Automating grid search for optimal parameters (e.g. `n_estimators` for Random Forest).
- **Collecting New Measurements:** Testing the model against real-world newly gathered physical flowers.
- **Deployment as a Web Application:** Building a Flask/FastAPI interface to predict species via a dashboard.
