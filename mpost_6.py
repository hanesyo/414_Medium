import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Sleep_Efficiency.csv')
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

df_clean = df.dropna(subset=['Sleep efficiency', 'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency'])
print(f"\nAfter cleaning: {len(df_clean)} rows\n")

def classify_sleep(efficiency):
    if efficiency >= 0.85:
        return 'Good Sleep'
    elif efficiency >= 0.65:
        return 'Average Sleep'
    else:
        return 'Poor Sleep'

df_clean['Sleep Quality'] = df_clean['Sleep efficiency'].apply(classify_sleep)

print("Sleep Quality Distribution:")
print(df_clean['Sleep Quality'].value_counts())
print()

df_clean['Smoking'] = df_clean['Smoking status'].map({'Yes': 1, 'No': 0})

feature_cols = [
    'Age',
    'Sleep duration',
    'REM sleep percentage',
    'Deep sleep percentage',
    'Light sleep percentage',
    'Awakenings',
    'Caffeine consumption',
    'Alcohol consumption',
    'Smoking',
    'Exercise frequency'
]

X = df_clean[feature_cols]
y = df_clean['Sleep Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training: {len(X_train)} | Test: {len(X_test)}\n")

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.1%}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['Good Sleep', 'Average Sleep', 'Poor Sleep'])
cm_df = pd.DataFrame(cm, index=['Good', 'Average', 'Poor'], columns=['Good', 'Average', 'Poor'])
print(cm_df)
print()

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance:")
print(feature_importance.to_string(index=False))
print()

test_df = df_clean.loc[X_test.index].copy()
test_df['Predicted'] = y_pred
test_df['Actual'] = y_test.values
misclassified = test_df[test_df['Predicted'] != test_df['Actual']].head(5)

print(f"Misclassifications: {len(test_df[test_df['Predicted'] != test_df['Actual']])} / {len(test_df)}\n")

if len(misclassified) > 0:
    for idx, row in misclassified.iterrows():
        print(f"ID {int(row['ID'])}: {row['Actual']} → {row['Predicted']} (Efficiency: {row['Sleep efficiency']:.2f})")
        print(f"  Age {int(row['Age'])}, Sleep {row['Sleep duration']:.1f}h, Deep {row['Deep sleep percentage']:.0f}%, Caffeine {row['Caffeine consumption']:.0f}mg")
        print()

results_df = test_df[['ID', 'Age', 'Gender', 'Actual', 'Predicted', 'Sleep efficiency'] + feature_cols]
results_df.to_csv('sleep_quality_results.csv', index=False)
print("Saved: sleep_quality_results.csv")
