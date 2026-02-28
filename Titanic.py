import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv("Titanic Dataset/train.csv")
test = pd.read_csv("Titanic Dataset/test.csv")
gender_submission = pd.read_csv("Titanic Dataset/gender_submission.csv")


print("Aperçu de l'ensemble d'entraînement :")
print(train.head())
print("\nInformations générales :")
print(train.info())


def preprocess(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df = df.drop(['Cabin', 'Name', 'Ticket'], axis=1)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

train_processed = preprocess(train)
test_processed = preprocess(test)


passenger_ids = test_processed["PassengerId"]


train_processed.drop("PassengerId", axis=1, inplace=True)
test_processed.drop("PassengerId", axis=1, inplace=True)


X = train_processed.drop("Survived", axis=1)
y = train_processed["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


tree_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_gini.fit(X_train, y_train)
y_pred_gini = tree_gini.predict(X_val)


tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_entropy.fit(X_train, y_train)
y_pred_entropy = tree_entropy.predict(X_val)


log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_logreg = log_reg.predict(X_val)


acc_gini = accuracy_score(y_val, y_pred_gini)
acc_entropy = accuracy_score(y_val, y_pred_entropy)
acc_logreg = accuracy_score(y_val, y_pred_logreg)

print("\n--- Arbre Gini ---\nAccuracy :", acc_gini)
print(classification_report(y_val, y_pred_gini))

print("\n--- Arbre Entropy ---\nAccuracy :", acc_entropy)
print(classification_report(y_val, y_pred_entropy))

print("\n--- Régression Logistique ---\nAccuracy :", acc_logreg)
print(classification_report(y_val, y_pred_logreg))


results = pd.DataFrame({
    'Modèle': ['Arbre Gini', 'Arbre Entropy', 'Régression Logistique'],
    'Précision': [acc_gini, acc_entropy, acc_logreg]
})

sns.barplot(data=results, x='Modèle', y='Précision', palette='coolwarm')
plt.title("Comparaison des modèles")
plt.ylim(0.6, 1)
plt.show()


plt.figure(figsize=(18, 8))
plot_tree(tree_gini, filled=True, feature_names=X.columns, class_names=['Non-survivant', 'Survivant'])
plt.title("Arbre de Décision - Gini")
plt.show()

plt.figure(figsize=(18, 8))
plot_tree(tree_entropy, filled=True, feature_names=X.columns, class_names=['Non-survivant', 'Survivant'])
plt.title("Arbre de Décision - Entropy")
plt.show()


best_model = tree_entropy  
test_predictions = best_model.predict(test_processed)


submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": test_predictions
})

submission.to_csv("titanic_submission.csv", index=False)
print("\n✅ Fichier de soumission 'titanic_submission.csv' créé.")
