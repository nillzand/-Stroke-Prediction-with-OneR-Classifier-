import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
from sklearn.metrics import  precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('stroke_prediction.csv')

# Drop the 'id' column
data.drop('id', axis=1, inplace=True)

# Fill missing values with median or mean
data['bmi'].fillna(data['bmi'].median(), inplace=True)
data['avg_glucose_level'].fillna(data['avg_glucose_level'].mean(), inplace=True)

# Replace 'Unknown' values in the 'smoking_status' column with the most frequent value
most_frequent = data[data['smoking_status'] != 'Unknown']['smoking_status'].mode()[0]
data['smoking_status'].replace('Unknown', most_frequent, inplace=True)

# Convert numerical data to categorical data
data['bmi_category'] = pd.cut(data['bmi'], bins=5, labels=['Underweight', 'Normal weight', 'Overweight', 'Obese I', 'Obese II'])
data['glucose_level_category'] = pd.qcut(data['avg_glucose_level'], q=3, labels=['Low', 'Medium', 'High'])

# Convert age to categorical data
data['age_category'] = pd.cut(data['age'], bins=[0, 18, 30, 45, 60, 120], labels=['<18', '18-30', '30-45', '45-60', '60+'])

# Drop the original numerical columns
data.drop(['bmi', 'avg_glucose_level', 'age'], axis=1, inplace=True)

# Split the data into features and target
X = data.drop('stroke', axis=1)
y = data['stroke']

# Define the model
class OneR(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        self.rules_ = []
        
    def fit(self, X, y):
        for feature in X.columns:
            feature_values = X[feature].unique()
            rules = {}
            for value in feature_values:
                label = Counter(y[X[feature] == value]).most_common(1)[0][0]
                error = sum((X[feature] == value) & (y != label))
                rules[value] = (label, error)
            best_rule = min(rules.items(), key=lambda x: x[1][1])
            self.rules_.append((feature, best_rule[0], best_rule[1][0]))
    
    def predict(self, X):
        predictions = []
        for i, row in X.iterrows():
            prediction = 1
            for rule in self.rules_:
                if row[rule[0]] == rule[1]:
                    prediction = rule[2]
                    break
            predictions.append(prediction)
        return predictions

# Set the number of splits
n_splits = 5

# Create KFold object using the number of splits
kf = KFold(n_splits=n_splits)

# Create lists to store accuracy, precision, recall, and F1-score for each test fold
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []


# For each test fold
for train_index, test_index in kf.split(X):
    # Extract training and test data from X and y using the indices created
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Create an instance of the model
    model = OneR()
    
    # Train the model using the training data
    model.fit(X_train, y_train)
    
    # Predict labels for the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy, precision, recall, and F1-score for the current test fold
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average='macro'))
    recall_list.append(recall_score(y_test, y_pred, average='macro'))
    f1_score_list.append(f1_score(y_test, y_pred, average='macro'))

# Calculate MicroAverage and MacroAverage measures
micro_average_accuracy = sum(accuracy_list) / n_splits
macro_average_accuracy = sum(accuracy_list) / len(accuracy_list)

micro_average_precision = sum(precision_list) / n_splits
macro_average_precision = sum(precision_list) / len(precision_list)

micro_average_recall = sum(recall_list) / n_splits
macro_average_recall = sum(recall_list) / len(recall_list)

micro_average_f1_score = sum(f1_score_list) / n_splits
macro_average_f1_score = sum(f1_score_list) / len(f1_score_list)

# Print MicroAverage and MacroAverage measures
print('MicroAverage Accuracy:', micro_average_accuracy)
print('MacroAverage Accuracy:', macro_average_accuracy)

print('MicroAverage Precision:', micro_average_precision)
print('MacroAverage Precision:', macro_average_precision)

print('MicroAverage Recall:', micro_average_recall)
print('MacroAverage Recall:', macro_average_recall)

print('MicroAverage F1-Score:', micro_average_f1_score)
print('MacroAverage F1-Score:', macro_average_f1_score)