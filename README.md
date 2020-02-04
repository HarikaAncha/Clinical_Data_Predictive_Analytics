# Content: Tempus
## Project: DScasestudy
The project is to investigate and analyze the attached data (DScasestudy.txt.gz), producing
a model that can be used to predict the “response”.
The first column of the provided data is the binary variable “response”. The 16,562 other
columns are binary columns that can be used to predict the “response”.

### Install

This project requires **Python 3** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [SMOTE](https://imbalanced-learn.readthedocs.io/)

All the above packages are can be installed with given Requirements.txt.

Used Google style docstrings.

### Code
Complete code implemented for this project is present in a single file models.py.


### Project 
Developed two models:
Model 1: Used SMOTE to make the dataset balanced and implemented a Random Forest Classifier model.
Model 2: Using the same unbalanced dataset developed the Logistic regression model with Lasso regularization.

The detailed project approach is explained in the DScasestudy Take-Home Documentation.




