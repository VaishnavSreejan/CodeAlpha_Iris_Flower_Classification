# Internship Explanation Guide

*Use this document as a cheat-sheet for explaining your project simply and confidently!*

### What is classification?
Classification is a type of supervised machine learning where the goal is to predict a category or class for a given data point. Instead of predicting a continuous number (like temperature or price), you predict a label (like "Spam" or "Not Spam", or in our case, the specific species of a flower).

### What is the Iris dataset?
It is a famous introductory dataset in Data Science. It contains 150 records of Iris flowers, fifty each from three different species. For each flower, it records four measurements of its petals and sepals. 

### What are features?
Features are the independent variables or input data that we feed into our model. In this project, the four features are Sepal Length, Sepal Width, Petal Length, and Petal Width.

### What is a target?
The target is the dependent variable we are trying to predict. Here, the target is the `species` of the flower (Setosa, Versicolor, or Virginica).

### Why train-test split?
If we test the model on the exact same data it used to learn (train), it's like giving a student the exam paper to study the night before. By doing a train-test split, we hide a portion of the data (20%) during training. We then use this hidden data to test the model, giving us a realistic idea of how well it will perform in the real world on totally unseen flowers.

### What is feature scaling?
Features often have different ranges (e.g. one might range from 0 to 1, while another ranges from 10 to 100). If we don't scale them, distance-based models (like KNN) might incorrectly assume that the feature with the larger numbers is drastically more important. Scaling harmonizes all features to a common scale.

### What is Logistic Regression?
Despite the word "regression", it's a classification algorithm. It uses a mathematical function (the sigmoid function) to estimate the probability that a data point belongs to a certain class.

### What is KNN (K-Nearest Neighbors)?
It's an intuitive algorithm. If we want to classify a new flower, we look at the 'K' (e.g., 5) flowers in our training data that have the most similar measurements (the nearest neighbors). Whatever species is most common among those 5 neighbors is the species we predict for the new flower.

### What is a Decision Tree?
It works like a massive flowchart. The model asks a series of True/False questions about the features (e.g., "Is Petal Length < 2.45cm?"). It follows the answers down the tree branches until it reaches a final classification leaf.

### What is Random Forest?
This is an "ensemble" algorithm. Instead of creating one Decision Tree, it creates dozens or hundreds of them, each trained on a slightly different random subset of the data. When making a prediction, all the trees "vote" and the majority wins. This fixes the risk of a single Decision Tree overfitting.

### What is accuracy?
The fundamental metric: Total Correct Predictions divided by Total Predictions made. If I test 30 flowers and get 27 right, my accuracy is 90%.

### What is precision?
Out of all the flowers the model *claimed* were Setosa, how many were *actually* Setosa? It measures the quality of a positive prediction.

### What is recall?
Out of all the truly actual Setosa flowers that existed in the test dataset, how many did the model successfully find? 

### What is F1-score?
Sometimes Precision and Recall are in conflict (improving one lowers the other). The F1-score is the Harmonic Mean of Precision and Recall. It provides one balancing metric to summarize both.

### What is a confusion matrix?
It’s a grid/table showing where the model gets confused. The rows represent the actual correct species, and columns representing what the model predicted. High numbers on the diagonal mean correct predictions.

### Why was the final model selected?
We evaluate the models on the test set using the F1-score. The selected model is the one that achieved the highest metric without showing signs of severe overfitting. 
