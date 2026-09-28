"""
Check SNR of all 8 Mojito-light catalog sources with both TDI generations:

    tdi_gen=1 -> EqualArmlengthOrbits (analytic), 1st-generation TDI, AET channels
    tdi_gen=2 -> ESAOrbits (numerical), 2nd-generation TDI, AE channels

(see GWfuncs_noise.build_waveform_response / GravWaveAnalysis).

Catalog -> waveform-parameter mapping (verified by reproducing the hardcoded
EMRI_G (ID=7) and EMRI_C (ID=3) params used in map_mojito_light.py /
map_mojito_light_emri_c.py):

    m1, m2      = PrimaryMassSSBFrame, SecondaryMassSSBFrame
    a           = PrimarySpinParameter
    p0          = SemiLatusRectum
    e0          = Eccentricity
    xI0         = 1.0  (equatorial-only waveform model)
    dist        = LuminosityDistance / 1000  (Mpc -> Gpc)
    qS, phiS    = icrs_to_ecliptic(pi/2 - Declination, RightAscension)
    qK, phiK    = icrs_to_ecliptic(PolarAnglePrimarySpin, AzimuthalAnglePrimarySpin)
    Phi_phi0    = AzimuthalPhase
    Phi_theta0  = PolarPhase
    Phi_r0      = RadialPhase

No Tobs/TDI-gen convention is documented anywhere for the catalog's
EstimatedSNR column, so this script just reports our own SNR at a fixed
(T, dt) for both TDI generations side by side with EstimatedSNR, rather than
trying to reproduce it exactly.
"""
import os
import sys
import csv
import pickle
import traceback
import faulthandler

faulthandler.enable(file=sys.stdout)

import numpy as np
import few

dir_work = '/home/svu/e1498138/emri_search/work/'
os.chdir(dir_work)
sys.path.insert(0, dir_work)

from GWfuncs_noise import GravWaveAnalysis, build_waveform_response

cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")

use_gpu = True
dt = 5
T = 2.0  # years; generous upper bound, waveform/response truncate at plunge

CATALOG_PATH = os.path.join(dir_work, 'emri_cat_mojito_lite_processed_MT.csv')


def icrs_to_ecliptic(qK_icrs, phiK_icrs, eps_deg=23.43929111):
    """
    Convert a sky direction (colatitude, longitude) from the ICRS/equatorial
    frame to the ecliptic frame (SSB frame used by the LISA response,
    is_ecliptic_latitude=False convention), via the standard
    equatorial<->ecliptic rotation with J2000 mean obliquity eps.
    Matches map_mojito_light_emri_c.py's icrs_to_ecliptic.
    """
    dec = np.pi / 2 - qK_icrs
    ra = phiK_icrs
    eps = np.deg2rad(eps_deg)

    sin_beta = np.sin(dec) * np.cos(eps) - np.cos(dec) * np.sin(ra) * np.sin(eps)
    beta = np.arcsin(sin_beta)

    y = np.cos(dec) * np.sin(ra) * np.cos(eps) + np.sin(dec) * np.sin(eps)
    x = np.cos(dec) * np.cos(ra)
    lam = np.arctan2(y, x) % (2 * np.pi)

    qK_ecl = np.pi / 2 - beta
    phiK_ecl = lam
    return qK_ecl, phiK_ecl


def load_catalog(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def row_to_params(row):
    m1 = float(row['PrimaryMassSSBFrame'])
    m2 = float(row['SecondaryMassSSBFrame'])
    a = float(row['PrimarySpinParameter'])
    p0 = float(row['SemiLatusRectum'])
    e0 = float(row['Eccentricity'])
    xI0 = 1.0
    dist = float(row['LuminosityDistance']) / 1000.0  # Mpc -> Gpc

    qS_icrs = np.pi / 2 - float(row['Declination'])
    phiS_icrs = float(row['RightAscension'])
    qS, phiS = icrs_to_ecliptic(qS_icrs, phiS_icrs)

    qK, phiK = icrs_to_ecliptic(
        float(row['PolarAnglePrimarySpin']), float(row['AzimuthalAnglePrimarySpin'])
    )

    Phi_phi0 = float(row['AzimuthalPhase'])
    Phi_theta0 = float(row['PolarPhase'])
    Phi_r0 = float(row['RadialPhase'])

    return [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0]


print(f"Using dt={dt}s, T={T}yr")
print('Building tdi_gen=1 (EqualArmlengthOrbits) response + analysis...')
resp_g1 = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=1)
gwf_g1 = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=1)

print('Building tdi_gen=2 (ESAOrbits) response + analysis...')
resp_g2 = build_waveform_response(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=2)
gwf_g2 = GravWaveAnalysis(T=T, dt=dt, use_gpu=use_gpu, tdi_gen=2)

rows = load_catalog(CATALOG_PATH)

results = []
for row in rows:
    src_id = row['ID']
    params = row_to_params(row)
    m1, m2 = params[0], params[1]
    catalog_snr = float(row['EstimatedSNR'])

    print(f"\n--- Source ID={src_id}  m1={m1:.3g}  m2={m2:.3g} ---", flush=True)

    try:
        print("  building h1 (tdi_gen=1)...", flush=True)
        h1 = resp_g1(*params)
        snr1 = float(gwf_g1.rhostat(h1))
    except Exception:
        print(f"  tdi_gen=1 failed:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        snr1 = float('nan')

    try:
        print("  building h2 (tdi_gen=2)...", flush=True)
        h2 = resp_g2(*params)
        snr2 = float(gwf_g2.rhostat(h2))
    except Exception:
        print(f"  tdi_gen=2 failed:", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        snr2 = float('nan')

    print(f"  EstimatedSNR (catalog) = {catalog_snr:.4g}")
    print(f"  SNR tdi_gen=1 (EqualArmlength, AET) = {snr1:.4g}")
    print(f"  SNR tdi_gen=2 (ESAOrbits, AE)       = {snr2:.4g}")

    results.append({
        'ID': src_id,
        'params': params,
        'EstimatedSNR': catalog_snr,
        'SNR_tdi_gen1': snr1,
        'SNR_tdi_gen2': snr2,
    })

print("\n\n=== Summary ===")
print(f"{'ID':>3} {'EstimatedSNR':>13} {'SNR_gen1':>10} {'SNR_gen2':>10}")
for r in results:
    print(f"{r['ID']:>3} {r['EstimatedSNR']:>13.4g} {r['SNR_tdi_gen1']:>10.4g} {r['SNR_tdi_gen2']:>10.4g}")

out_path = os.path.join(dir_work, 'search', 'check_snr_mojito_all.pkl')
with open(out_path, 'wb') as f:
    pickle.dump({'T': T, 'dt': dt, 'results': results}, f)
print(f"\nSaved to {out_path}")
