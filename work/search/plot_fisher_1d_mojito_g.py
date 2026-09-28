import pickle

import numpy as np
import matplotlib.pyplot as plt

with open('/home/svu/e1498138/emri_search/work/fisher_mojito_light_g.pkl', 'rb') as f:
    d = pickle.load(f)

cov = d['cov']
theta_true = d['theta_true_scaled']
sigmas = np.sqrt(np.diag(cov))

labels = [r'$\log_{10} m_1$', r'$\log_{10} m_2$', r'$a$', r'$p_0$', r'$e_0$',
          r'$\cos q_S$', r'$\phi_S$']
ndim = len(labels)

fig_1d, axs_1d = plt.subplots(1, ndim, figsize=(4 * ndim, 4))
for dim in range(ndim):
    ax = axs_1d[dim]
    mu, sig = theta_true[dim], sigmas[dim]

    x = np.linspace(mu - 5 * sig, mu + 5 * sig, 400)
    gauss = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    ax.plot(x, gauss, '-', color='blue', alpha=0.7, linewidth=2, label='Fisher Gaussian')

    ax.axvline(mu, color='red', linestyle='--', alpha=0.5, label='True Point')

    ax.set_xlabel(labels[dim], fontsize=12)
    ax.grid(True, alpha=0.3)

axs_1d[0].set_ylabel('pdf', fontsize=12)
axs_1d[0].legend(fontsize=9)
fig_1d.tight_layout()
fig_1d.savefig('/home/svu/e1498138/emri_search/work/search/fisher_mojito_light_g_1d.png',
                dpi=150, bbox_inches='tight')
print('Saved fisher_mojito_light_g_1d.png')
