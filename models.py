from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import decomposition
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import auc, classification_report, mean_squared_error, roc_curve, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Loading data

data_df = pd.read_csv(
    'C:\\Users\\Ancha Harika\\Desktop\\tempus\\DScasestudy.txt', sep="\t")
data_df.head()
data_df.tail()
data_df.describe()


# Data Preparation

# Checking for missing values
data_df.isnull().sum().sum()
# Removing duplicates
reduced_data_df = data_df.T.drop_duplicates(keep='first').T
reduced_data_df.drop(columns=['V4', 'V6'])
# Separating response from features
X_variables = reduced_data_df.drop(['response'], axis=1)
response = reduced_data_df[['response']]
# Dividing the data into training and testing data sets 60/40
x_train, x_test, y_train, y_test = train_test_split(
    X_variables, response, test_size=0.4, shuffle=True, random_state=999)
# Dividing data into testing and training data sets 70/30
x_train_30, x_test_30, y_train_30, y_test_30 = train_test_split(
    X_variables, response, test_size=0.3, shuffle=True, random_state=999)
# Dividing data into testing and training data sets 60---40
x_train_20, x_test_20, y_train_20, y_test_20 = train_test_split(
    X_variables, response, test_size=0.2, shuffle=True, random_state=999)
# Using ggplot
plt.style.use('ggplot')
x = ['0', '1']
unique_values_frequency = [407, 123]
x_pos = [i for i, _ in enumerate(x)]
# Plotting the frequency of 1's and 0's in response column
plt.bar(x_pos, unique_values_frequency, color='green')
plt.xlabel("1's and 0's")
plt.ylabel("Frequency")
plt.title("Bar Chart 1's and 0's")
plt.xticks(x_pos, x)
plt.show()

# MODEL 1  (RandomForestClassifier)


def model_train_error(X_train, y_train, model):
    ''' Caluculates the training error of the given model

        Args:
        X_train(dataframe) : feature training data
        y_train(dataframe) : response training data
        model : model implemented 

        Returns:
        mse : mean squared error
    '''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    return mse


def model_validation_error(X_test, y_test, model):
    ''' Caluculates the testing error of the given model

        Args:
        X_test(dataframe) : feature testing data
        y_test(dataframe) : response testing data
        model : model implemented 

        Returns:
        mse : mean squared error
    '''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse


def errors(trainig_x, training_y, testing_x, testing_y, your_model):
    ''' Prints Testing and Training error's

        Args:
        trainig_x(dataframe) : feature training data
        training_y(dataframe) : response training data
        testing_x(dataframe) : feature testing data
        testing_y(dataframe) : response testing data
        your_model : model implemented 

    '''
    print("train_error", model_train_error(trainig_x, training_y, your_model))
    print("test_error", model_validation_error(
        testing_x, testing_y, your_model))


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
class_weight = ['balanced']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'class_weight': class_weight}

rf = RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
    estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
    random_state=99, n_jobs=-1)
# Fit the RandomForestClassifier model 60/40
sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train['response'])
rf_40 = rf_random.fit(x_train_res, y_train_res)
# Predict model
# Using the forest's predict method on the test data
predictions_40 = rf_40.predict(x_test)
print("Confusion Matrix", confusion_matrix(y_test, predictions_40))
print(classification_report(y_test, predictions_40))
print("mean squared error", mean_squared_error(y_test, predictions_40))
print("accuracy", accuracy_score(y_test, predictions_40))
errors(x_train_res, y_train_res, x_test, y_test, rf_random)
# Fit the RandomForestClassifier model 70/30
x_train_res_30, y_train_res_30 = sm.fit_sample(
    x_train_30, y_train_30['response'])
rf_30 = rf_random.fit(x_train_res_30, y_train_res_30)
# Predict model
# Using the forest's predict method on the test data
predictions_30 = rf_30.predict(x_test_30)
print("Confusion Matrix", confusion_matrix(y_test_30, predictions_30))
print(classification_report(y_test_30, predictions_30))
print("mean squared error", mean_squared_error(y_test_30, predictions_30))
print("accuracy", accuracy_score(y_test_30, predictions_30))
errors(x_train_res_30, y_train_res_30, x_test_30, y_test_30, rf_random)
# Fit the RandomForestClassifier model
x_train_res_20, y_train_res_20 = sm.fit_sample(
    x_train_20, y_train_20['response'])
rf_20 = rf_random.fit(x_train_res_20, y_train_res_20)
# Predict model
# Using the forest's predict method on the test data
predictions_20 = rf_20.predict(x_test_20)
print("Confusion Matrix", confusion_matrix(y_test_20, predictions_20))
print(classification_report(y_test_20, predictions_20))
print("mean squared error", mean_squared_error(y_test_20, predictions_20))
print("accuracy", accuracy_score(y_test_20, predictions_20))
errors(x_train_res_20, y_train_res_20, x_test_20, y_test_20, rf_random)

#           Model-1 validation

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_variables, response, shuffle=True, test_size=0.2, random_state=999)
# train/validation split (gives us train and validation sets)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_val,
                                                                y_train_val,
                                                                shuffle=False,
                                                                test_size=0.25,
                                                                random_state=99)
rfs = rf_random.fit(X_train, y_train)
# print proportions
print(
    'train: {}% ; validation: {}% ; test {}%'.format(
        round(len(y_train) / len(response),
              2),
        round(len(y_validation) / len(response),
              2),
        round(len(y_test) / len(response),
              2)))
# calculate errors
new_train_error = mean_squared_error(y_train, rfs.predict(X_train))
new_validation_error = mean_squared_error(
    y_validation, rfs.predict(X_validation))
new_test_error = mean_squared_error(y_test, rfs.predict(X_test))
print("new train error", new_train_error)
print("new test error", new_test_error)
print("new validation error", new_validation_error)

#        Model-1's 5-fold cross validation

sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=999)
X_res, y_res = sm.fit_resample(X_variables, response['response'])
print("number of unique values", y_res.nunique())
unique_values_count = y_res.value_counts()
print("value counts of unique values\n", unique_values_count)
K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=10000)
train_errors = []
validation_errors = []
accuracy = []
auc_array = []
for train_index, val_index in kf.split(X_res, y_res):
    # Split data
    X_train, X_val = X_res.iloc[train_index], X_res.iloc[val_index]
    y_train, y_val = y_res.iloc[train_index], y_res.iloc[val_index]
    # Instantiate model
    model_k = rf_random.fit(X_train, y_train)
    # Predict model
    predictions_kfold = model_k.predict(X_val)
    print("Confusion Matrix", confusion_matrix(y_val, predictions_kfold))
    print(classification_report(y_val, predictions_kfold))
    print("accuracy_score", accuracy_score(y_val, predictions_kfold))
    accuracy.append(accuracy_score(y_val, predictions_kfold))
    predict_prob = model_k.predict_proba(X_val)
    fpr, tpr, _ = roc_curve(y_val, predict_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.show()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    k = auc(fpr, tpr)
    print("auc", k)
    auc_array.append(k)
    # Calculate errors
    train_error = model_train_error(X_train, y_train, model_k)
    val_error = model_validation_error(X_val, y_val, model_k)
    # Append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
print("train_error mean", sum(train_errors)/len(train_errors))
print("validation_error mean", sum(validation_errors)/len(validation_errors))
print("accuracy_mean", sum(accuracy)/len(accuracy))
print("Auc_mean", sum(auc_array)/len(auc_array))

#    MODEL 2 (Logistic Regression with Lasso Regularization)

# Dividing data into testing and training data sets 40/60

x_train, x_test, y_train, y_test = train_test_split(
    X_variables, response, test_size=0.4, shuffle=True, random_state=999)
log = LogisticRegression(penalty='l1', solver='saga',
                         random_state=999, max_iter=10000, tol=1e-07, n_jobs=1)
log_40 = log.fit(x_train, y_train)
# Predict Model
# Using Logistic predict model
prediction_40 = log_40.predict(x_test)
print(confusion_matrix(y_test, prediction_40))
print(classification_report(y_test, prediction_40))
print("mean squared error", mean_squared_error(y_test, prediction_40))
print("accuracy", accuracy_score(y_test, prediction_40))
errors(x_train, y_train, x_test, y_test, log_40)


# Dividing data into testing and training data sets 30/70

x_train_30, x_test_30, y_train_30, y_test_30 = train_test_split(
    X_variables, response, test_size=0.3, shuffle=True, random_state=999)
log_30 = log.fit(x_train_30, y_train_30)
# Predict Model
# Using Logistic predict model
prediction_30 = log_30.predict(x_test_30)
confusion_matrix(y_test_30, prediction_30)
print(confusion_matrix(y_test_30, prediction_30))
print(classification_report(y_test_30, prediction_30))
print("mean squared error", mean_squared_error(y_test_30, prediction_30))
print("accuracy", accuracy_score(y_test_30, prediction_30))
errors(x_train_30, y_train_30, x_test_30, y_test_30, log_30)

# Dividing data into testing and training data sets 40/60

x_train_20, x_test_20, y_train_20, y_test_20 = train_test_split(
    X_variables, response, test_size=0.2, shuffle=True, random_state=999)
log_20 = log.fit(x_train_20, y_train_20)
# Predict Model
# Using Logistic predict model
prediction_20 = log.predict(x_test_20)
confusion_matrix(y_test_20, prediction_20)
print(confusion_matrix(y_test_20, prediction_20))
print(classification_report(y_test_20, prediction_20))
print("mean squared error", mean_squared_error(y_test_20, prediction_20))
print("accuracy", accuracy_score(y_test_20, prediction_20))
errors(x_train_20, y_train_20, x_test_20, y_test_20, log_20)

#      Model-2 validation

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_variables, response, shuffle=True, test_size=0.2, random_state=999)
# train/validation split (gives us train and validation sets)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_val,
                                                                y_train_val,
                                                                shuffle=False,
                                                                test_size=0.25,
                                                                random_state=99)

log_fit = log.fit(X_train, y_train)
print(
    'train: {}% ; validation: {}% ; test {}%'.format(
        round(len(y_train) / len(response),
              2),
        round(len(y_validation) / len(response),
              2),
        round(len(y_test) / len(response),
              2)))
# calculate errors
# Predict Model
new_train_error = mean_squared_error(y_train, log_fit.predict(X_train))
new_validation_error = mean_squared_error(
    y_validation, log_fit.predict(X_validation))
new_test_error = mean_squared_error(y_test, log_fit.predict(X_test))
print("new train error", new_train_error)
print("new test error", new_test_error)
print("new validation error", new_validation_error)

#       Model-2's 5-fold cross validation

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=10000)
train_errors = []
validation_errors = []
accuracy = []
auc_array = []
for train_index, val_index in kf.split(X_variables, response):
    # Split data
    X_train, X_val = X_variables.iloc[train_index], X_variables.iloc[val_index]
    y_train, y_val = response.iloc[train_index], response.iloc[val_index]
    # Instantiate model
    model_k = log.fit(X_train, y_train)
    # Predict Model
    predictions_kfold = model_k.predict(X_val)
    print("Confusion Matrix", confusion_matrix(y_val, predictions_kfold))
    print(classification_report(y_val, predictions_kfold))
    print("accuracy_score", accuracy_score(y_val, predictions_kfold))
    accuracy.append(accuracy_score(y_val, predictions_kfold))
    predict_prob = model_k.predict_proba(X_val)
    fpr, tpr, _ = roc_curve(y_val, predict_prob[:, 1])
    plt.plot(fpr, tpr)
    plt.show()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    k = auc(fpr, tpr)
    print("auc", k)
    auc_array.append(k)
    # Calculate errors
    train_error = model_train_error(X_train, y_train, model_k)
    val_error = model_validation_error(X_val, y_val, model_k)
    # Append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
