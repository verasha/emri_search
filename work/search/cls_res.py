import os
import sys
import pickle
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D
import parismc
from parismc import Sampler

def prior_transform(u):
    u = np.asarray(u)
    logm1lim, logm2lim, alim, p0lim, e0lim = [5.6, 6.4], [0.8, 1.3], [0.3, 0.99], [8.0, 11.0], [0.2, 0.5]
    if u.ndim == 1: u_2d = u.reshape(1, -1)
    else: u_2d = u
    t_2d = np.zeros_like(u_2d)
    t_2d[:, 0] = (logm1lim[1] - logm1lim[0]) * u_2d[:, 0] + logm1lim[0]
    t_2d[:, 1] = (logm2lim[1] - logm2lim[0]) * u_2d[:, 1] + logm2lim[0]
    t_2d[:, 2] = (alim[1] - alim[0]) * u_2d[:, 2] + alim[0]
    t_2d[:, 3] = (p0lim[1] - p0lim[0]) * u_2d[:, 3] + p0lim[0]
    t_2d[:, 4] = (e0lim[1] - e0lim[0]) * u_2d[:, 4] + e0lim[0]
    return t_2d.flatten() if u.ndim == 1 else t_2d

def log_density(x): return 0.0

# Load data
with open('/scratch/e1498138/paris1_sf/int_1yr_s12/sampler_state.pkl', 'rb') as f:
    sampler = pickle.load(f)

samples, weights = sampler.get_samples_with_weights(flatten=True)
try:
    log_densities = np.concatenate([sampler.searched_log_densities_list[j][:sampler.element_num_list[j]] for j in range(sampler.n_proc)])
except:
    log_densities = np.zeros(len(samples))

total_samples = len(samples)
# Strictly locked user-specified ranges
user_ranges = [
    (5.6, 6.4),    # logM1
    (0.8, 1.3),    # logM2
    (0.3, 0.99),   # a
    (8.0, 11.0),   # p0
    (0.2, 0.5),    # e0
]
true_params = np.array([6.0, 1.0, 0.7, 9.0, 0.4])
labels = ["logM1", "logM2", "a", "p0", "e0"]
_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
markers_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']

# ---------------------------------------------------------
# Common plotting utilities
# ---------------------------------------------------------
def generate_base_corner(pts_to_fit_range, title):
    # Strictly use the specified user_ranges; no longer dynamically expand based on point positions
    fig = corner.corner(np.vstack([pts_to_fit_range, [true_params]]), 
                        labels=labels, 
                        range=user_ranges,
                        plot_datapoints=False, 
                        plot_density=False, 
                        plot_contours=False,
                        hist_kwargs=dict(alpha=0, lw=0))
    fig.suptitle(title, fontsize=10)
    return fig

# ---------------------------------------------------------
# 1. Clustering plot logic
# ---------------------------------------------------------
def get_best_points_per_cluster(all_samples, all_values, n_clusters=10, top_n_pool=100):
    pool_size = min(top_n_pool, len(all_samples))
    pool_indices = np.argsort(all_values)[-pool_size:]
    pool_samples = all_samples[pool_indices]
    pool_values = all_values[pool_indices]
    km = KMeans(n_clusters=n_clusters, max_iter=20, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(pool_samples)
    best_indices_in_pool = []
    for i in range(n_clusters):
        in_cluster = (cluster_labels == i)
        if not np.any(in_cluster): continue
        best_idx_in_cluster = np.argmax(pool_values[in_cluster])
        best_indices_in_pool.append(pool_indices[in_cluster][best_idx_in_cluster])
    best_indices_in_pool = np.array(best_indices_in_pool)
    final_sort = np.argsort(all_values[best_indices_in_pool])[::-1]
    return best_indices_in_pool[final_sort]

def generate_cluster_plot(indices, filename, title):
    print(f"Generating Plot (Strict Range): {filename}...")
    top_pts = samples[indices]
    top_lds = log_densities[indices]
    
    fig = generate_base_corner(top_pts, title)
    
    # Find the closest point using normalized distance
    norm_diff = (top_pts - true_params) / np.array([r[1]-r[0] for r in user_ranges])
    distances = np.linalg.norm(norm_diff, axis=1)
    closest_idx = np.argmin(distances)

    # Print all points, marking the closest one
    print(f"--- Points for {filename} ---")
    print(f"    True params: " + ", ".join(f"{l}={v:.4f}" for l, v in zip(labels, true_params)))
    for i, pt in enumerate(top_pts):
        tag = " <-- CLOSEST" if i == closest_idx else ""
        pt_str = ", ".join(f"{l}={v:.4f}" for l, v in zip(labels, pt))
        print(f"    pt {i+1}: {pt_str} | ld={top_lds[i]:.4f} | dist={distances[i]:.4f}{tag}")
    cp = top_pts[closest_idx]
    cp_str = ", ".join(f"{l}={v:.4f}" for l, v in zip(labels, cp))
    print(f"    >> Closest point: pt {closest_idx+1}: {cp_str} | ld={top_lds[closest_idx]:.4f} | dist={distances[closest_idx]:.4f}")

    for i, pt in enumerate(top_pts):
        corner.overplot_points(fig, pt.reshape(1, -1), color=_cycle[i % len(_cycle)], marker=markers_list[i % len(markers_list)], ms=8)
    corner.overplot_points(fig, np.array([true_params]), color='black', marker='+', ms=12)
    
    # Enforce axis range limits again to ensure subplot coordinates are strictly consistent
    axes = np.array(fig.axes).reshape((5, 5))
    for r in range(5):
        for c in range(5):
            if c <= r: axes[r, c].set_xlim(user_ranges[c])
            if c < r: axes[r, c].set_ylim(user_ranges[r])

    handles = []
    for i in range(len(top_pts)):
        lbl = f"pt {i+1} (ld={top_lds[i]:.2f})"
        if i == closest_idx: lbl += " [CLOSEST]"
        handles.append(Line2D([0],[0], color=_cycle[i % len(_cycle)], marker=markers_list[i % len(markers_list)], ls='', ms=7, label=lbl))
    handles.append(Line2D([0],[0], color='black', marker='+', ls='', ms=8, label='True Params'))
    
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=6)
    plt.savefig(filename, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------
# 2. Averages plot logic
# ---------------------------------------------------------
def generate_averages_plot(n_list, filename, title):
    print(f"Generating Averages Plot (Strict Range): {filename}...")
    sorted_indices = np.argsort(log_densities)[::-1]
    
    avg_pts = []
    avg_lds = []
    
    for n in n_list:
        idx = sorted_indices[:n]
        avg_pts.append(np.mean(samples[idx], axis=0))
        avg_lds.append(np.mean(log_densities[idx]))
    
    avg_pts = np.array(avg_pts)
    fig = generate_base_corner(avg_pts, title)
    
    for i, pt in enumerate(avg_pts):
        corner.overplot_points(fig, pt.reshape(1, -1), color=_cycle[i % len(_cycle)], marker=markers_list[i % len(markers_list)], ms=9)
    corner.overplot_points(fig, np.array([true_params]), color='black', marker='+', ms=12)

    axes = np.array(fig.axes).reshape((5, 5))
    for r in range(5):
        for c in range(5):
            if c <= r: axes[r, c].set_xlim(user_ranges[c])
            if c < r: axes[r, c].set_ylim(user_ranges[r])
    
    handles = []
    for i, n in enumerate(n_list):
        lbl = f"Avg Top {n} (avg_ld={avg_lds[i]:.2f})"
        handles.append(Line2D([0],[0], color=_cycle[i % len(_cycle)], marker=markers_list[i % len(markers_list)], ls='', ms=8, label=lbl))
    handles.append(Line2D([0],[0], color='black', marker='+', ls='', ms=8, label='True Params'))
    
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=7)
    plt.savefig(filename, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
print(f"=== Report: total number of samples = {total_samples:,} ===")

# 1. Cluster analysis plots (Top 100/1000)
best_ld_100 = get_best_points_per_cluster(samples, log_densities, n_clusters=10, top_n_pool=100)
best_ld_1000 = get_best_points_per_cluster(samples, log_densities, n_clusters=10, top_n_pool=1000)
generate_cluster_plot(best_ld_100, 'corner_cluster_best_logden_top100.png', "Top 100 Clusters (Best Log-Den)")
generate_cluster_plot(best_ld_1000, 'corner_cluster_best_logden_top1000.png', "Top 1000 Clusters (Best Log-Den)")

# 2. Global Top 10 plot
global_top10 = np.argsort(log_densities)[-10:][::-1]
generate_cluster_plot(global_top10, 'corner_global_top10_logden.png', "Global Top 10 Log-Density Samples")

# 3. Averages comparison plot
n_values = [10, 20, 50, 100, 500, 1000]
generate_averages_plot(n_values, 'corner_top_n_averages.png', "Averages of Top-N Log-Density Samples")

print("\nTask complete. All plots have been locked to the strict coordinate ranges.")
