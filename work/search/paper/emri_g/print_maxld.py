#!/usr/bin/env python
"""Print the MAP (max log-density) point per process from a parismc sampler_state.pkl.

Usage:
    .venv/bin/python print_maxld.py [savepath]

savepath defaults to the stage2_merge_brute directory; pass a different
directory (containing sampler_state.pkl) to point at another run.
"""
import os
import sys
import numpy as np
import parismc

DIR_SCRATCH = '/scratch/e1498138/'
DEFAULT_SAVEPATH = DIR_SCRATCH + 'paper/ext/emri_g/stage2_merge_brute/'

LOGM1_LIM = [5.81854, 5.83713]
LOGM2_LIM = [1.99115, 1.99507]
A_LIM     = [0.49296, 0.51048]
P0_LIM    = [15.53112, 15.88282]
E0_LIM    = [0.74301, 0.74536]
DIST_LIM  = [3,  5]
COSQS_LIM = [-1.0, 1.0]
PHIS_LIM  = [0.0,  2 * np.pi]
COSQK_LIM = [-1.0, 1.0]
PHIK_LIM  = [0.0,  2 * np.pi]
PHIPHI0_LIM = [0.0, 2 * np.pi]
PHIR0_LIM   = [0.0, 2 * np.pi]


def prior_transform(u):
    t = np.zeros_like(u)
    t[:, 0]  = (LOGM1_LIM[1] - LOGM1_LIM[0]) * u[:, 0] + LOGM1_LIM[0]
    t[:, 1]  = (LOGM2_LIM[1] - LOGM2_LIM[0]) * u[:, 1] + LOGM2_LIM[0]
    t[:, 2]  = (A_LIM[1] - A_LIM[0]) * u[:, 2] + A_LIM[0]
    t[:, 3]  = (P0_LIM[1] - P0_LIM[0]) * u[:, 3] + P0_LIM[0]
    t[:, 4]  = (E0_LIM[1] - E0_LIM[0]) * u[:, 4] + E0_LIM[0]
    t[:, 5]  = (DIST_LIM[1] - DIST_LIM[0]) * u[:, 5] + DIST_LIM[0]
    t[:, 6]  = (COSQS_LIM[1] - COSQS_LIM[0]) * u[:, 6] + COSQS_LIM[0]
    t[:, 7]  = (PHIS_LIM[1] - PHIS_LIM[0]) * u[:, 7] + PHIS_LIM[0]
    t[:, 8]  = (COSQK_LIM[1] - COSQK_LIM[0]) * u[:, 8] + COSQK_LIM[0]
    t[:, 9]  = (PHIK_LIM[1] - PHIK_LIM[0]) * u[:, 9] + PHIK_LIM[0]
    t[:, 10] = (PHIPHI0_LIM[1] - PHIPHI0_LIM[0]) * u[:, 10] + PHIPHI0_LIM[0]
    t[:, 11] = (PHIR0_LIM[1] - PHIR0_LIM[0]) * u[:, 11] + PHIR0_LIM[0]
    return t


def main():
    savepath = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAVEPATH
    print(f"Loading {savepath}/sampler_state.pkl ...")
    sampler = parismc.Sampler.load_state(savepath + '/sampler_state.pkl')

    proc_pt = sampler.searched_points_list
    logden_list = sampler.searched_log_densities_list
    n_proc = len(sampler.now_covariances)
    n_filled = sampler.element_num_list  # actual filled length per proc (rest is padding)

    out_dir = os.path.join(savepath, 'maxld_pts')
    os.makedirs(out_dir, exist_ok=True)

    maxld_pts = []
    for i in range(n_proc):
        pts_i = proc_pt[i][:n_filled[i]]
        ld_i = logden_list[i][:n_filled[i]]
        best_idx = np.nanargmax(ld_i)
        pt = prior_transform(pts_i[best_idx].reshape(1, -1))
        maxld_pts.append(pt)
        globals()[f"maxld_pt{i + 1}"] = pt  # maxld_pt1, maxld_pt2, ...

        np.save(os.path.join(out_dir, f"maxld_pt{i + 1}.npy"), pt)
        print(f"proc {i}: idx={best_idx}  max_ld={ld_i[best_idx]:.5f}  pt={pt.ravel()}"
              f"  -> saved {out_dir}/maxld_pt{i + 1}.npy")

    np.savez(os.path.join(out_dir, "maxld_pts_all.npz"),
             **{f"maxld_pt{i + 1}": p for i, p in enumerate(maxld_pts)})
    print(f"\nAll points also saved together: {out_dir}/maxld_pts_all.npz")

    return maxld_pts


if __name__ == '__main__':
    main()
