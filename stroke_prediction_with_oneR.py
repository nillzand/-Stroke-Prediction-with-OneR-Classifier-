import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('stroke', axis=1), data['stroke'], test_size=0.2, random_state=42)


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



# Train the model
model = OneR()
model.fit(X_train, y_train)

# Test the model and report accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Print best rule
best_rule = model.rules_[0]
print('Best rule:', best_rule)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)

# Calculate and print precision, recall, and F-measure
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Precision:', precision)
print('Recall:', recall)
print('F-measure:', f1)