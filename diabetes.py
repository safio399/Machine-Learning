import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('diabetes.csv')


print(df.head())




df.hist(bins=20, figsize=(12, 10))
plt.tight_layout()
plt.show()


sns.pairplot(df, hue="Outcome")
plt.show()




X = df.drop("Outcome", axis=1)
y = df["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)


acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_lr = accuracy_score(y_test, y_pred_lr)

print("Decision Tree Accuracy:", acc_dt)
print("Random Forest Accuracy:", acc_rf)
print("Logistic Regression Accuracy:", acc_lr)

models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
accuracies = [acc_dt, acc_rf, acc_lr]

plt.figure(figsize=(8,6))
sns.barplot(x=models, y=accuracies)
plt.ylabel("Accuracy")
plt.ylim(0.7, 1)
plt.title("Comparaison des modèles")
plt.show()
