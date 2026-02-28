import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


digits = load_digits()
X = digits.data
y = digits.target


print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)
print("Valeurs uniques :", set(y))


plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X[i].reshape(8, 8), cmap="gray")
    plt.title(f"Étiquette : {y[i]}")
    plt.axis('off')
plt.suptitle("Quelques images du dataset MNIST (version sklearn)")
plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tree_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_gini.fit(X_train, y_train)
y_pred_gini = tree_gini.predict(X_test)


tree_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
tree_entropy.fit(X_train, y_train)
y_pred_entropy = tree_entropy.predict(X_test)


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)


acc_gini = accuracy_score(y_test, y_pred_gini)
acc_entropy = accuracy_score(y_test, y_pred_entropy)
acc_logreg = accuracy_score(y_test, y_pred_logreg)

print("\n--- Arbre Gini ---")
print("Accuracy :", acc_gini)

print("\n--- Arbre Entropy ---")
print("Accuracy :", acc_entropy)

print("\n--- Régression Logistique ---")
print("Accuracy :", acc_logreg)


results = pd.DataFrame({
    'Modèle': ['Arbre Gini', 'Arbre Entropy', 'Régression Logistique'],
    'Précision': [acc_gini, acc_entropy, acc_logreg]
})

sns.barplot(data=results, x="Modèle", y="Précision", palette="Set2")
plt.ylim(0.8, 1)
plt.title("Comparaison des précisions sur le dataset MNIST")
plt.show()

ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, cmap="Blues")
plt.title("Matrice de confusion - Régression Logistique")
plt.show()


