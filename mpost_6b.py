import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Sleep_Efficiency.csv')
df_clean = df.dropna(subset=['Sleep efficiency', 'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency']).copy()

def classify_sleep(efficiency):
    if efficiency >= 0.85:
        return 'Good Sleep'
    elif efficiency >= 0.65:
        return 'Average Sleep'
    else:
        return 'Poor Sleep'

df_clean['Sleep Quality'] = df_clean['Sleep efficiency'].apply(classify_sleep)
df_clean['Smoking'] = df_clean['Smoking status'].map({'Yes': 1, 'No': 0})

feature_cols = ['Age', 'Sleep duration', 'REM sleep percentage', 'Deep sleep percentage', 
                'Light sleep percentage', 'Awakenings', 'Caffeine consumption', 
                'Alcohol consumption', 'Smoking', 'Exercise frequency']

X = df_clean[feature_cols]
y = df_clean['Sleep Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(y_test, y_pred, labels=['Good Sleep', 'Average Sleep', 'Poor Sleep'])
cm_df = pd.DataFrame(cm, index=['Good', 'Average', 'Poor'], columns=['Good', 'Average', 'Poor'])
print(cm_df)
print()

print("FEATURE IMPORTANCE")
print("="*50)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))
