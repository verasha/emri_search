"""
Compute the per-mode power distribution across individual (l, m, n) Teukolsky
modes for two points:
  - the recovered point:  [5.82595158, 1.99267966, 0.49587826, 15.73817815,
                            0.74416625, 0.82400395, 3.55596108]
  - the true (injected) point, from the Mojito-light EMRI_G source in
    emri_g_stage2_f.ipynb

Uses GravWaveAnalysis.calc_power (GWfuncs_noise.py), the same routine used in
work/old/check_power.ipynb: |h_lmn|^2 (with the -m modes reconstructed via
conjugate symmetry through m0mask), noise-weighted by the PSD evaluated at
each mode's own instantaneous GW frequency along the trajectory, summed over
the trajectory. This is NOT the ModeSelector.SNR_approx phase-gated
cross-mode inner product used for mode *selection* in modeselectoralt.py.

Both points are in the 7-d sampled parameter space
[log10(m1), log10(m2), a, p0, e0, cos(qS), phiS]; the remaining nuisance
parameters (xI0, dist, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0) are fixed at
the injected source values.

Saves a CSV table and a bar plot per point under ./mode_power_distribution/.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dir_work = "/home/svu/e1498138/emri_search/work/"
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response
from few.utils.constants import YRSID_SI
from few.utils.geodesic import get_fundamental_frequencies

# ---------------------------------------------------------------- settings
use_gpu = True
tdi_gen = 1
dt = 5
T = 14 / 12  # near plunge -- source plunges ~2 months after 1yr
N_traj = 5000

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mode_power_distribution_6mth")
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------- fixed source
# Mojito light EMRI_G (nuisance params fixed at injected values)
xI0 = 1.0
dist = 4.755
qK = 0.9327436243181905
phiK = 3.6550014190179883
Phi_phi0 = 3.1940
Phi_theta0 = 3.3780
Phi_r0 = 0.6038

m1_true, m2_true = 6.72e5, 9.84e1
a_true, p0_true, e0_true = 0.5, 15.7117, 0.7440
qS_true, phiS_true = 0.5906, 3.5808

point_true = [np.log10(m1_true), np.log10(m2_true), a_true, p0_true, e0_true,
              np.cos(qS_true), phiS_true]
# maxld_pt1 -- the original recovered point
point_pt1 = [5.82595158, 1.99267966, 0.49587826, 15.73817815,
             0.74416625, 0.82400395, 3.55596108]
# maxld_pt9 -- the recovered point
point_pt9 = [5.82618268, 1.99290506, 0.49938496, 15.73408894,
             0.74390361, 0.82196702, 3.59040418]

points = {"true": point_true, "pt1": point_pt1, "pt9": point_pt9}  # , 


def to_full_params(point):
    """[log10(m1), log10(m2), a, p0, e0, cos(qS), phiS] -> full 14-param vector."""
    logm1, logm2, a, p0, e0, cosqS, phiS = point
    qS = np.arccos(cosqS)
    return [10**logm1, 10**logm2, a, p0, e0, xI0, dist, qS, phiS,
            qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]


# ------------------------------------------------------------------- build
print("Building ResponseWrapper...")
waveform_response = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

print("Building GravWaveAnalysis...")
gwf = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=tdi_gen)

inner_gen = waveform_response.waveform_gen.waveform_generator
traj = getattr(inner_gen, "inspiral_generator", None)
amp = getattr(inner_gen, "amplitude_generator", None)
ylm_gen = getattr(inner_gen, "ylm_gen", None)

delta_T = T * YRSID_SI / N_traj


def _get_viewing_angles(qS, phiS, qK, phiK):
    """Transform from the detector frame to the source frame."""
    cqS, sqS = np.cos(qS), np.sin(qS)
    cphiS, sphiS = np.cos(phiS), np.sin(phiS)
    cqK, sqK = np.cos(qK), np.sin(qK)
    cphiK, sphiK = np.cos(phiK), np.sin(phiK)
    R = np.array([sqS * cphiS, sqS * sphiS, cqS])
    S = np.array([sqK * cphiK, sqK * sphiK, cqK])
    phi = -np.pi / 2.0
    theta = np.arccos(-np.dot(R, S))
    return theta, phi


def mode_power_distribution(params):
    """
    Compute the per-mode power of every individual (l, m, n) Teukolsky mode
    (including the -m modes reconstructed via conjugate symmetry) for the
    given full 14-param point, via GravWaveAnalysis.calc_power.

    Returns a list of dicts, sorted by descending power fraction, each with
    keys: l, m, n, power, fraction (percent of the summed per-mode power
    Sum_lmn calc_power(l,m,n) -- calc_power isn't in the same units as
    rho^2 = gwf.inner(h_fft, h_fft) (no delta_T integration measure, no
    distance factor), so fractions are relative to the per-mode total,
    not to the true signal SNR^2).
    """
    (m1, m2, a, p0, e0, xI0_, dist_, qS, phiS, qK_, phiK_,
     Phi_phi0_, Phi_theta0_, Phi_r0_) = params

    theta, phi = _get_viewing_angles(qS, phiS, qK_, phiK_)

    _, p, e, x, _, _, _ = traj(m1, m2, a, p0, e0, xI0_, T=T, dt=delta_T,
                                upsample=True, Phi_phi0=Phi_phi0_,
                                Phi_theta0=Phi_theta0_, Phi_r0=Phi_r0_)

    teuk_modes = amp(a, p, e, x)
    ylms = ylm_gen(amp.unique_l, amp.unique_m, theta, phi).copy()[amp.inverse_lm]

    OmegaPhi, _, OmegaR = get_fundamental_frequencies(a, p, e, x)

    # Use the _no_mask arrays consistently -- these are what teuk_modes and
    # m0mask are actually indexed by. (amp.l_arr/m_arr/n_arr can end up being
    # a different, already-expanded ±m array depending on prior calls made on
    # this amp instance elsewhere, which silently misaligns pos_labels/
    # gw_freqs against m0mask/teuk_modes -- see mode power distribution bug
    # writeup.)
    l_arr = amp.l_arr_no_mask.get() if hasattr(amp.l_arr_no_mask, "get") else np.asarray(amp.l_arr_no_mask)
    m_arr = amp.m_arr_no_mask.get() if hasattr(amp.m_arr_no_mask, "get") else np.asarray(amp.m_arr_no_mask)
    n_arr = amp.n_arr_no_mask.get() if hasattr(amp.n_arr_no_mask, "get") else np.asarray(amp.n_arr_no_mask)

    m0mask = amp.m_arr_no_mask != 0
    m0mask_cpu = m0mask.get() if hasattr(m0mask, "get") else m0mask

    # gw_freqs must match calc_power's own concatenation order in full_modes:
    # teuk_modes' native (+m) columns first, then the reconstructed -m
    # conjugate columns for every m0mask==True entry, in the same order.
    # Each reconstructed -m column needs its OWN (mirrored) frequency, not
    # the +m one -- omitting this silently PSD-weights every -m mode at the
    # wrong frequency (a ~2*m*Omega_phi error), which is large enough to
    # flip which mode looks dominant.
    # NOTE: the -m reconstruction is Z_{l,-m,-n} = (-1)^l * conj(Z_{l,m,n})
    # (Drasco & Hughes, Eq. 86) -- BOTH m and n flip together, not just m.
    # So the reconstructed mode's frequency is simply the negative of the
    # original stored mode's frequency (omega_{-m,-n} = -omega_{m,n}), the
    # standard negative-frequency mirror needed to build a real waveform
    # from a one-sided complex spectrum.
    gw_frequencies_per_mode = []
    for idx in range(len(l_arr)):
        m = m_arr[idx].get() if hasattr(m_arr[idx], "get") else m_arr[idx]
        n = n_arr[idx].get() if hasattr(n_arr[idx], "get") else n_arr[idx]
        gw_frequencies_per_mode.append(m * OmegaPhi + n * OmegaR)
    for idx in range(len(l_arr)):
        if not m0mask_cpu[idx]:
            continue
        m = m_arr[idx].get() if hasattr(m_arr[idx], "get") else m_arr[idx]
        n = n_arr[idx].get() if hasattr(n_arr[idx], "get") else n_arr[idx]
        gw_frequencies_per_mode.append(-m * OmegaPhi + n * OmegaR)

    total_power = gwf.calc_power(teuk_modes, ylms, m0mask, m1=m1, m2=m2,
                                  gw_freqs=gw_frequencies_per_mode)
    total_power = total_power.get() if hasattr(total_power, "get") else total_power

    # matches the concat order in calc_power: positive-m modes as stored,
    # followed by the conjugated -m modes for every m0mask==True entry.
    #
    # amp.n_arr_no_mask's own sign convention for n is the OPPOSITE of what
    # ModeSelector/mode_selection use elsewhere in this codebase -- confirmed
    # empirically: waveform_response(mode_selection=[(2,-2,-3)]) reproduces
    # the large rho this script's amp.n_arr labels as n=+3, and vice versa.
    # Negating n uniformly here (both halves) was checked against the
    # independently-confirmed matched-filter ranking (via mode_selection
    # rho spot-checks) and matches it much more closely than the alternative
    # "flip n only for the reconstructed half" derivation, which was tried
    # and reverted -- it collapsed n=-2's reported power to ~0.2%, despite
    # mode_selection=(2,-2,-2) independently confirming n=-2 carries
    # substantial power. Treat this as empirically-validated, not a fully
    # closed-form derivation of the underlying convention.
    pos_labels = [(int(l), int(m), -int(n)) for l, m, n in zip(l_arr, m_arr, n_arr)]
    neg_labels = [(int(l), -int(m), -int(n)) for l, m, n, keep
                  in zip(l_arr, m_arr, n_arr, m0mask_cpu) if keep]
    mode_labels = pos_labels + neg_labels

    power_sum = float(np.sum(total_power))

    records = [
        {"l": l, "m": m, "n": n, "power": float(p),
         "fraction": 100.0 * float(p) / power_sum}
        for (l, m, n), p in zip(mode_labels, total_power)
    ]
    records.sort(key=lambda r: r["fraction"], reverse=True)
    return records, power_sum


# ------------------------------------------------------------------- run
results = {}
for name, point in points.items():
    print(f"\n=== {name} point ===")
    params = to_full_params(point)
    records, power_sum = mode_power_distribution(params)
    results[name] = records

    captured = sum(r["fraction"] for r in records)
    print(f"sum of per-mode power = {power_sum:.6g}, captured by (l,m,n) grid = {captured:.4f}%")
    print(f"{'l':>3} {'m':>4} {'n':>4} {'power':>14} {'fraction %':>12}")
    for r in records[:20]:
        print(f"{r['l']:>3} {r['m']:>4} {r['n']:>4} {r['power']:>14.6g} {r['fraction']:>12.4f}")

    # CSV
    csv_path = os.path.join(out_dir, f"mode_power_{name}.csv")
    with open(csv_path, "w") as f:
        f.write("l,m,n,power,fraction_pct\n")
        for r in records:
            f.write(f"{r['l']},{r['m']},{r['n']},{r['power']:.8g},{r['fraction']:.6f}\n")
    print(f"Saved {csv_path}")

    # bar plot, top 20 modes
    top = records[:20]
    fig, ax = plt.subplots(figsize=(10, 4))
    tick_labels = [f"({r['l']},{r['m']},{r['n']})" for r in top]
    ax.bar(range(len(top)), [r["fraction"] for r in top])
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(tick_labels, rotation=90)
    ax.set_ylabel("% of total power")
    ax.set_title(f"Mode power distribution ({name} point)")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"mode_power_{name}.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved {png_path}")

# -------------------------------------------------------------- comparison
top_true = results["true"][:15]
labels = [f"({r['l']},{r['m']},{r['n']})" for r in top_true]
frac_true = [r["fraction"] for r in top_true]
x = np.arange(len(labels))
width = 0.4

for other_name in [n for n in points if n != "true"]:
    fig, ax = plt.subplots(figsize=(11, 4))
    lookup_other = {(r["l"], r["m"], r["n"]): r["fraction"] for r in results[other_name]}
    frac_other = [lookup_other.get((r["l"], r["m"], r["n"]), 0.0) for r in top_true]

    ax.bar(x - width / 2, frac_true, width, label="true")
    ax.bar(x + width / 2, frac_other, width, label=other_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("% of total power")
    ax.set_title(f"Mode power distribution: true vs {other_name}")
    ax.legend()
    fig.tight_layout()
    cmp_path = os.path.join(out_dir, f"mode_power_comparison_{other_name}.png")
    fig.savefig(cmp_path)
    plt.close(fig)
    print(f"\nSaved {cmp_path}")
