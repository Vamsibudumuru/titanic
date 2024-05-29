import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(df.head())

# Display dataset info
print(df.info())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Fill missing values for 'age' with the median age
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing values for 'embarked' with the most common value
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop rows with missing 'embark_town' and 'deck' as these are not crucial for our analysis
df.drop(columns=['embark_town', 'deck'], inplace=True)

# Drop the 'alive' column as it is redundant with 'survived'
df.drop(columns=['alive'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
df['sex'] = label_encoder.fit_transform(df['sex'])
df['embarked'] = label_encoder.fit_transform(df['embarked'])
df['class'] = label_encoder.fit_transform(df['class'])

# Drop columns that won't be used in the model
df.drop(columns=['who', 'adult_male', 'embark_town', 'alone'], inplace=True)

# Define features (X) and target (y)
X = df.drop('survived', axis=1)
y = df['survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Display classification report
class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for visualization
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(features_df)

# Plot the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importances')
plt.show()
