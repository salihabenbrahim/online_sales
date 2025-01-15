import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

data = pd.read_csv('Online_Sales_Data.csv')


cleaned_data = data.drop(['Transaction ID', 'Date', 'Product Name'], axis=1)

# Encode categorical variables
label_encoders = {}
for column in ['Product Category', 'Region', 'Payment Method']:
    le = LabelEncoder()
    cleaned_data[column] = le.fit_transform(cleaned_data[column])
    label_encoders[column] = le


target = 'Payment Method'
X = cleaned_data.drop(target, axis=1)
y = cleaned_data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Model selection and training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 3. Model evaluation
y_pred = model.predict(X_test)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 4. Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoders['Payment Method'].classes_, yticklabels=label_encoders['Payment Method'].classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Output results
print("Classification Report:")
print(class_report)

# Graph: Impact of Payment Methods on Sales
payment_impact = data.groupby('Payment Method')['Total Revenue'].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
payment_impact.plot(kind='bar', color='skyblue')
plt.title('Impact of Payment Methods on Total Sales')
plt.xlabel('Payment Method')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Identification des produits les plus vendus dans chaque catégorie
top_products = data.groupby(['Product Category', 'Product Name']).sum(numeric_only=True)['Units Sold']
top_products = top_products.sort_values(ascending=False).groupby(level=0).head(1)
print("Produits les plus vendus par catégorie :\n", top_products)

