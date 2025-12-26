import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def ols(x, y):
    X = np.column_stack((np.ones(x.shape[0]), x))
    coef = la.inv(X.T @ X) @ X.T @ y
    return coef

def phi_i_exp(x,mu_i,sigma_i):
    return (1+np.exp(-((x - mu_i))/sigma_i))**(-1)

def ols_phi_x(x,y,q):
    X_tmp = np.zeros((len(x),q))
    for i in range (0,len(x)):
        for j in range (1,q+1):
            X_tmp[i][j-1]=x[i]**(j)
    X = np.column_stack((np.ones(x.shape[0]), X_tmp))
    coef = la.inv(X.T @ X) @ X.T @ y
    print("Coefficient shape: ")
    print(coef.shape)
    return coef , X_tmp

import numpy as np
import matplotlib.pyplot as plt

def ols(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    beta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    beta0 = y_mean - beta1 * x_mean
    return beta0, beta1


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# TRAITEMENT DATA 1
data1 = np.load("data1.npy")
x1, y1 = data1[0, :], data1[1, :]
b0_1, b1_1 = ols(x1, y1)
droite1 = b0_1 + b1_1 * x1
err1 = np.mean((y1 - droite1)**2)

ax1.scatter(x1, y1)
ax1.plot(x1, droite1, color='red')
ax1.set_title("Data 1 : Modèle Linéaire Pertinent")
ax1.legend()

# TRAITEMENT DATA 2 
data2 = np.load("data2.npy")
x2, y2 = data2[0, :], data2[1, :]
b0_2, b1_2 = ols(x2, y2)
droite2 = b0_2 + b1_2 * x2
err2 = np.mean((y2 - droite2)**2)

ax2.scatter(x2, y2)
ax2.plot(x2, droite2,color='red')
ax2.set_title("Data 2 : Modèle Non-Linéaire (Erreur élevée)")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Erreur d'apprentissage Data 1: {err1}")
print(f"Erreur d'apprentissage Data 2: {err2}")
import numpy as np
import matplotlib.pyplot as plt

# Création de la figure avec 2 colonnes pour comparer q=10
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# QUESTION 4 : DATA 2 (q=10)
data2 = np.load("data2.npy")
x2, y2 = data2[0, :], data2[1, :]
coef2, _ = ols_phi_x(x2, y2, 10)
# Calcul de la courbe
x_range2 = np.linspace(min(x2), max(x2), 200)
y_pred2 = coef2[0] + sum(coef2[j] * (x_range2**j) for j in range(1, len(coef2)))

ax1.scatter(x2, y2)
ax1.plot(x_range2, y_pred2, color='red')
ax1.set_title("Data 2 : Ajustement q=10")
ax1.legend()
R2 = np.mean((y2 - (coef2[0] + sum(coef2[j] * (x2**j) for j in range(1, len(coef2)))))**2)

# --- QUESTION 5 : DATA 3 (q=10) ---
data3 = np.load("data3.npy")
x3, y3 = data3[0, :], data3[1, :]
coef3, _ = ols_phi_x(x3, y3, 10)
# Calcul de la courbe
x_range3 = np.linspace(min(x3), max(x3), 200)
y_pred3 = coef3[0] + sum(coef3[j] * (x_range3**j) for j in range(1, len(coef3)))

ax2.scatter(x3, y3)
ax2.plot(x_range3, y_pred3, color='red')
ax2.set_title("Data 3 : Surapprentissage (Overfitting) q=10")
ax2.legend()
R3 = np.mean((y3 - (coef3[0] + sum(coef3[j] * (x3**j) for j in range(1, len(coef3)))))**2)

plt.tight_layout()
plt.show()

print(f"Erreur d'apprentissage Data 2 (q=10) : {R2}")
print(f"Erreur d'apprentissage Data 3 (q=10) : {R3}")


#question 7 

def ridge_poly(x, y, q, lambda_):
    X_tmp = np.zeros((len(x), q))
    for i in range(len(x)):
        for j in range(q):
            X_tmp[i, j] = x[i] ** (j+1)
    X = np.column_stack((np.ones(len(x)), X_tmp))
    I = np.eye(X.shape[1])
    I[0, 0] = 0 
    coef = la.inv(X.T @ X + lambda_ * I) @ X.T @ y
    return coef, X

data3 = np.load("data3.npy")
x = data3[0, :]
y = data3[1, :]
lambdas = [0, 0.01, 0.1, 1, 10, 100]

plt.figure(figsize=(12, 7))
plt.scatter(x, y, label="Data", color="black", zorder=3)

print("Résultats de la Régression Ridge (q=10) :")
print("-" * 40)

for lam in lambdas:
    coef, X = ridge_poly(x, y, 10, lam)
    y_pred = coef[0] + sum(coef[j] * (x**j) for j in range(1, len(coef)))
    mse = np.mean((y - y_pred)**2)
    print(f"Lambda = {lam:6} | Erreur = {mse:.5f}")
    x_fine = np.linspace(min(x), max(x), 500)
    y_fine = coef[0] + sum(coef[j] * (x_fine**j) for j in range(1, len(coef)))
    plt.plot(x_fine, y_fine, label=f"λ = {lam}")

plt.title("régularisation Ridge sur Data 3 (q=10)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.ylim(min(y)-0.5, max(y)+0.5) 
plt.grid(True, alpha=0.3)
plt.show()

#question 8 lasso

from sklearn.linear_model import Lasso

plt.figure(figsize=(12, 7))
plt.scatter(x, y, label="Data", color="black", zorder=3)
X_poly = np.zeros((len(x), 10))

for j in range(10):
    X_poly[:, j] = x ** (j + 1)

print("\nRésultats de la Régression LASSO (q=10) :")
print("-" * 40)
for lam in lambdas:
    model = Lasso(alpha=lam, max_iter=10000)
    
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    mse = np.mean((y - y_pred)**2)
    # Comptage des coefficients nuls (parcimonie)
    n_nuls = np.sum(model.coef_ == 0)
    print(f"Lambda = {lam:6} | Erreur = {mse:.5f} | Coeffs nuls = {n_nuls}/10")
    # Tracé de la courbe
    x_fine = np.linspace(min(x), max(x), 500)
    X_fine_poly = np.zeros((len(x_fine), 10))
    for j in range(10):
        X_fine_poly[:, j] = x_fine ** (j + 1)
    y_fine = model.predict(X_fine_poly)
    
    plt.plot(x_fine, y_fine, label=f"λ = {lam}")

plt.title("Régularisation LASSO sur Data 3 (q=10)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.ylim(min(y)-0.5, max(y)+0.5) 
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()