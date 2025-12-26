###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
plt.close('all')
###############################################################################

################################################################################
# PARAMETERS
################################################################################
# Dimension and sample size
p=2
n=600
# Proportion of sample from classes 0, 1, and outliers
p0 = [0.5,2.5/6,0.5/6]
p1 = [0.5,2.5/6,3/6]
pout = [0,1/6,2.5/6]

# Examples of means/covariances of classes 0, 1 and outliers
mu0 = np.array([-2,-2])
mu1 = np.array([2,2])
muout = np.array([-8,-8])
Sigma_ex1 = np.eye(p)
Sigma_ex2 = np.array([[5, 0.1],
                      [1, 0.5]])
Sigma_ex3 = np.array([[0.5, 1],
                      [1, 5]])
Sigma0 = Sigma_ex1
Sigma1 = Sigma_ex1
Sigmaout = Sigma_ex1
# Regularization coefficient
lamb = 10

################################################################################
# DATA/LABELS GENERATION
################################################################################
def generate(n,p0,p1,pout,mu0,mu1,muout,Sigma0,Sigma1,Sigmaout,p=2):
  n0 = int(np.floor(n*p0))
  n1 = int(np.floor(n*p1))
  nout = int(np.floor(n*pout))
  if n0+n1+nout < n:
    n0 += int(n - (n0+n1+nout))
  # Data and labels
  mu0_mat = mu0.reshape((p,1))@np.ones((1,n0))
  mu1_mat = mu1.reshape((p,1))@np.ones((1,n1))
  x0 = np.zeros((p,n0+nout))
  x0[:,0:n0] = mu0_mat + la.sqrtm(Sigma0)@rnd.randn(p,n0)
  x1 = mu1_mat + la.sqrtm(Sigma1)@rnd.randn(p,n1)
  if nout > 0:
    muout_mat = muout.reshape((p,1))@np.ones((1,nout))
    x0[:,n0:n0+nout] = muout_mat + la.sqrtm(Sigmaout)@rnd.randn(p,nout)
  y = np.concatenate((-np.ones(n0+nout),np.ones(n1)))
  X = np.ones((n,p+1))
  for i in np.arange(n):
      X[0:n0+nout,1:p+1] = x0.T
      X[n0+nout:n,1:p+1] = x1.T
  return X, y, x0, x1, n0, nout

################################################################################
def plot_hyperplan(ax, beta0,beta,label,style,X):
  X_data=X[:,1:]
  x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
  y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))
  Z = beta0 + beta[0] * xx + beta[1] * yy
  ax.contour(xx, yy, Z, levels=[0], colors=style, linestyles='dashed', label=label)


fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i in range(3):
    # Génération des données
    X, y, x0, x1, n0, nout = generate(n, p0[i], p1[i], pout[i], mu0, mu1, muout, Sigma0, Sigma1, Sigmaout, p)
    X_data = X[:, 1:]
    ax = axes[i]

    # Modèle OLS (Vert)
    ols_model = LinearRegression(fit_intercept=True)
    ols_model.fit(X_data, y)
    
    # Modèle Ridge (Orange)
    alpha_val = 500
    ridge_model = Ridge(alpha=alpha_val, fit_intercept=True)
    ridge_model.fit(X_data, y)

    # Modèle Logistique (Violet)
    log_reg = LogisticRegression(C=1/lamb, fit_intercept=True)
    log_reg.fit(X_data, y)

    #Affichage des points
    ax.plot(x0[0, :], x0[1, :], 'xb', label='Classe 0 (et outliers)')
    ax.plot(x1[0, :], x1[1, :], 'xr', label='Classe 1')

    #  Affichage des hyperplans
    plot_hyperplan(ax, ols_model.intercept_, ols_model.coef_, "OLS", "green", X)
    plot_hyperplan(ax, ridge_model.intercept_, ridge_model.coef_, "Ridge", "orange", X)
    plot_hyperplan(ax, log_reg.intercept_[0], log_reg.coef_[0], "Logistique", "purple", X)

    # Mise en forme
    ax.set_title(f"Test {i+1}")

plt.suptitle("Comparaison OLS vs Ridge vs Logistique face aux Outliers", fontsize=16, y=1.05)
plt.show()