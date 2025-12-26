###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
plt.close('all')
# Regularization coefficient
lamb = 1
###############################################################################
#1 PARTIE MNIST (À mettre après votre boucle for)
###############################################################################
# 1. Chargement des vraies données MNIST
print("Chargement de MNIST...")
mnist = fetch_openml(data_id=554, parser='auto')
X_mnist = mnist.data.to_numpy()
y_mnist = np.array(mnist.target, dtype=int)

# Division Train/Test 
X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.25, random_state=0)

# 2. Normalisation 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # On transforme le test avec les paramètres du train

# 3. Entraînement Logistique L2 
tol = 1e-4 # Tolérance plus stricte
clf = LogisticRegression(C=1/lamb, fit_intercept=True, penalty='l2', solver='lbfgs', max_iter=500, tol=tol)

print("Entraînement sur MNIST avec régularisation L2 en cours...")
clf.fit(X_train_scaled, y_train)

# 4. Évaluation et Matrice de Confusion 
y_pred = clf.predict(X_test_scaled)
cm_L2 = confusion_matrix(y_test, y_pred)

#plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_L2, display_labels=np.unique(y_mnist))
disp.plot(cmap='Blues', values_format='d')
plt.title("Matrice de Confusion : MNIST (Régression Logistique L2)")
plt.show()
###############################################################################
# 2. AFFICHAGE DES COEFFICIENTS SOUS FORME D'IMAGE
###############################################################################
# Les coefficients sont dans clf.coef_ (forme: k_classes, p_features) 
coefs = clf.coef_

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    # On reshape les 784 coefficients en une image 28x28
    coef_img = coefs[i].reshape(28, 28)
    
    # Rouge = poids positifs (favorise la classe), Bleu = poids négatifs 
    vmax = np.abs(coefs).max()
    plt.imshow(coef_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.title(f"Classe {i}")
    plt.axis('off')

plt.suptitle(r"Visualisation des coefficients $\hat{\beta}$ par classe ($L^2$)")
plt.show()
###############################################################################
# 3. RÉGULARISATION L1 (LASSO)
###############################################################################
# Le solveur 'saga' est indispensable pour L1.
# On garde C=1/lamb

clf_l1 = LogisticRegression(C=1/lamb, fit_intercept=True, penalty='l1', solver='saga', max_iter=1000, tol=tol)

print("Entraînement sur MNIST avec régularisation L1 en cours...")
clf_l1.fit(X_train_scaled, y_train)

# Affichage des coefficients L1
coefs_l1 = clf_l1.coef_

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    coef_img = coefs_l1[i].reshape(28, 28)
    
    # On utilise la même échelle de couleur pour comparer
    vmax = np.abs(coefs_l1).max()
    plt.imshow(coef_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
    plt.title(f"Classe {i} (L1)")
    plt.axis('off')

plt.suptitle(r"Visualisation des coefficients $\hat{\beta}$ par classe (Régularisation $L^1$)")
plt.show()