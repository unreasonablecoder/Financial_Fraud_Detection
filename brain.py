import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the cleaned data
df = pd.read_csv('cleaned_fraud_data.csv')

# 2. Separate Features (X) and Target (y)
X = df.drop('Class', axis=1) 
y = df['Class']

# 3. Split into Training (80%) and Testing (20%)
# 'stratify=y' ensures both sets have the same % of fraud
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the model... this may take a moment.")

# 4. Initialize Random Forest 
# 'class_weight=balanced' is the secret sauce for that 0.17% imbalance!
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Test the model
y_pred = model.predict(X_test)

# 6. The Results
print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))