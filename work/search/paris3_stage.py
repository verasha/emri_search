import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np

import few
from few.waveform import GenerateEMRIWaveform, FastKerrEccentricEquatorialFlux

os.chdir('/nfs/home/svu/e1498138/localgit/FEWNEW/work/')
sys.path.insert(0, '/nfs/home/svu/e1498138/localgit/FEWNEW/work/')

import GWfuncs
import loglike_pure
import loglike_timemax
import parismc


cfg_set = few.get_config_setter(reset=True)
cfg_set.set_log_level("info")


SEARCH_DIR = '/nfs/home/svu/e1498138/localgit/FEWNEW/work/search'
DEFAULT_OUTPUT_ROOT = os.path.join(SEARCH_DIR, 'paris3_stage_results')
PRECOMPUTED_1YR_PATH = os.path.join(SEARCH_DIR, 'precomputed_lhs_paris3_1yr_1e+05.pkl')

TRUE_INTRINSIC = np.array([6.0, 1.0, 0.7, 9.0, 0.4], dtype=float)
PARIS2_BEST_FIT = np.array([6.03302293, 1.12261954, 0.69265393, 9.04744717, 0.31691929], dtype=float)

BROAD_LO = np.array([5.6, 0.8, 0.3, 8.0, 0.2], dtype=float)
BROAD_HI = np.array([6.4, 1.3, 0.99, 11.0, 0.5], dtype=float)

XTRINSIC = {
    'xI0': 1.0,
    'dist': 1.8,
    'qS': np.pi,
    'phiS': 0.0,
    'qK': 0.0,
    'phiK': 0.0,
    'Phi_phi0': 0.4,
    'Phi_theta0': 0.0,
    'Phi_r0': 0.5,
}


@dataclass(frozen=True)
class StageSpec:
    name: str
    T: float
    objective: str
    num_iterations: int
    S_schedule: tuple
    stop_stable_iters: int
    alpha: int
    trail_size: int
    gamma: int
    init_u_sigma: float
    half_width_scale: float
    min_half_widths: tuple
    use_precomputed_seed: bool = False


DEFAULT_STAGES = (
    StageSpec(
        name='timemax_3m',
        T=3 / 12,
        objective='timemax',
        num_iterations=25000,
        S_schedule=(3.0, 10.0, 30.0, 100.0),
        stop_stable_iters=6000,
        alpha=int(1e5),
        trail_size=int(1e3),
        gamma=500,
        init_u_sigma=0.03,
        half_width_scale=1.0,
        min_half_widths=(0.18, 0.12, 0.16, 0.70, 0.08),
    ),
    StageSpec(
        name='timemax_6m',
        T=6 / 12,
        objective='timemax',
        num_iterations=20000,
        S_schedule=(3.0, 10.0, 30.0, 100.0),
        stop_stable_iters=5000,
        alpha=int(3e4),
        trail_size=int(1e3),
        gamma=500,
        init_u_sigma=0.025,
        half_width_scale=0.60,
        min_half_widths=(0.08, 0.06, 0.08, 0.35, 0.04),
    ),
    StageSpec(
        name='pure_1y',
        T=12 / 12,
        objective='pure',
        num_iterations=18000,
        S_schedule=(1.0, 3.0, 10.0, 30.0),
        stop_stable_iters=4500,
        alpha=int(1e4),
        trail_size=int(1e3),
        gamma=500,
        init_u_sigma=0.020,
        half_width_scale=0.45,
        min_half_widths=(0.03, 0.02, 0.05, 0.20, 0.015),
        use_precomputed_seed=True,
    ),
    StageSpec(
        name='pure_2y',
        T=24 / 12,
        objective='pure',
        num_iterations=15000,
        S_schedule=(1.0, 3.0, 10.0, 30.0),
        stop_stable_iters=3500,
        alpha=int(8e3),
        trail_size=int(1e3),
        gamma=500,
        init_u_sigma=0.015,
        half_width_scale=0.35,
        min_half_widths=(0.015, 0.015, 0.03, 0.12, 0.01),
    ),
)


def make_waveform_generators(T, dt):
    use_gpu = True
    force_backend = "cuda12x"

    inspiral_kwargs = {
        "func": 'KerrEccEqFlux',
        "DENSE_STEPPING": 0,
        "include_minus_m": False,
    }
    amplitude_kwargs = {"force_backend": force_backend}
    Ylm_kwargs = {"force_backend": force_backend}
    sum_kwargs_comb = {"force_backend": force_backend, "pad_output": True}
    sum_kwargs_sep = {"force_backend": force_backend, "pad_output": True, "separate_modes": True}

    waveform_gen_comb = GenerateEMRIWaveform(
        FastKerrEccentricEquatorialFlux,
        frame='detector',
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs_comb,
        use_gpu=use_gpu,
    )
    waveform_gen_sep = GenerateEMRIWaveform(
        FastKerrEccentricEquatorialFlux,
        frame='detector',
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs_sep,
        use_gpu=use_gpu,
    )
    gwf = GWfuncs.GravWaveAnalysis(T, dt)
    return waveform_gen_comb, waveform_gen_sep, gwf


def make_loglike(stage):
    waveform_gen_comb, waveform_gen_sep, gwf = make_waveform_generators(stage.T, dt=10)
    params_star = (
        1e6, 1e1, 0.7, 9.0, 0.4,
        XTRINSIC['xI0'], XTRINSIC['dist'], XTRINSIC['qS'], XTRINSIC['phiS'],
        XTRINSIC['qK'], XTRINSIC['phiK'], XTRINSIC['Phi_phi0'],
        XTRINSIC['Phi_theta0'], XTRINSIC['Phi_r0']
    )
    n_vals = np.arange(-1, 6)
    ell = 2

    if stage.objective == 'timemax':
        loglike_obj = loglike_timemax.LogLikeTimeMax(
            params_star,
            waveform_gen_comb,
            gwf,
            verbose=False,
            waveform_gen_sep=waveform_gen_sep,
            ell=ell,
            n_vals=n_vals,
            M_mode=None,
        )
    elif stage.objective == 'pure':
        loglike_obj = loglike_pure.LogLikePure(
            params_star,
            waveform_gen_comb,
            gwf,
            verbose=False,
            waveform_gen_sep=waveform_gen_sep,
            ell=ell,
            n_vals=n_vals,
            M_mode=None,
        )
    else:
        raise ValueError(f"Unknown objective: {stage.objective}")

    data_snr = float(gwf.rhostat(loglike_obj.signal))
    return loglike_obj, data_snr


def phys_to_theta(phys):
    logm1, logm2, a_i, p0_i, e0_i = phys
    return np.array([
        10 ** logm1,
        10 ** logm2,
        a_i,
        p0_i,
        e0_i,
        XTRINSIC['xI0'],
        XTRINSIC['dist'],
        XTRINSIC['qS'],
        XTRINSIC['phiS'],
        XTRINSIC['qK'],
        XTRINSIC['phiK'],
        XTRINSIC['Phi_phi0'],
        XTRINSIC['Phi_theta0'],
        XTRINSIC['Phi_r0'],
    ])


def evaluate_one(loglike_obj, phys):
    return float(loglike_obj(phys_to_theta(np.asarray(phys, dtype=float))))


def make_bounds(center, prev_half_widths, stage):
    center = np.asarray(center, dtype=float)
    if prev_half_widths is None:
        half_widths = np.maximum((BROAD_HI - BROAD_LO) / 2.0, np.asarray(stage.min_half_widths, dtype=float))
    else:
        half_widths = np.maximum(
            stage.half_width_scale * np.asarray(prev_half_widths, dtype=float),
            np.asarray(stage.min_half_widths, dtype=float),
        )

    lo = np.maximum(BROAD_LO, center - half_widths)
    hi = np.minimum(BROAD_HI, center + half_widths)
    # Ensure the seed is interior and every width is non-zero.
    hi = np.maximum(hi, lo + 1e-6)
    center = np.clip(center, lo + 1e-8, hi - 1e-8)
    half_widths = 0.5 * (hi - lo)
    return lo, hi, half_widths, center


def make_prior_functions(lo, hi):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    span = hi - lo

    def prior_transform(u):
        u = np.asarray(u, dtype=float)
        return lo + span * u

    def inverse_prior_transform(params):
        params = np.asarray(params, dtype=float)
        return (params - lo) / span

    return prior_transform, inverse_prior_transform


def make_log_density(loglike_obj, temper_state):
    def log_density(params):
        params = np.asarray(params, dtype=float)
        if params.ndim == 1:
            params = params.reshape(1, -1)
        out = np.empty(params.shape[0], dtype=float)
        for i, phys in enumerate(params):
            try:
                out[i] = evaluate_one(loglike_obj, phys) * temper_state['S']
            except Exception:
                out[i] = -np.inf
        return out

    return log_density


def maybe_precomputed_seed(loglike_obj, current_seed):
    if not os.path.exists(PRECOMPUTED_1YR_PATH):
        return current_seed, None

    with open(PRECOMPUTED_1YR_PATH, 'rb') as f:
        data = pickle.load(f)

    lhs_phys = np.asarray(data['lhs_phys'], dtype=float)
    lhs_log_densities = np.asarray(data['log_densities'], dtype=float)
    best_idx = int(np.argmax(lhs_log_densities))
    candidate = lhs_phys[best_idx]

    try:
        cand_ld = evaluate_one(loglike_obj, candidate)
        curr_ld = evaluate_one(loglike_obj, current_seed)
    except Exception:
        return current_seed, None

    if cand_ld > curr_ld:
        return candidate, {'candidate_ld': cand_ld, 'seed_ld': curr_ld, 'source': PRECOMPUTED_1YR_PATH}
    return current_seed, {'candidate_ld': cand_ld, 'seed_ld': curr_ld, 'source': PRECOMPUTED_1YR_PATH}


def get_best_point_and_logden(sampler, prior_transform):
    best_ld = -np.inf
    best_u = None
    for j in range(len(sampler.searched_points_list)):
        n = sampler.element_num_list[j]
        if n <= 0:
            continue
        logdens = sampler.searched_log_densities_list[j][:n]
        idx = int(np.argmax(logdens))
        if logdens[idx] > best_ld:
            best_ld = float(logdens[idx])
            best_u = sampler.searched_points_list[j][idx].copy()

    if best_u is None:
        raise RuntimeError("Sampler returned no points")

    best_phys = prior_transform(best_u.reshape(1, -1))[0]
    return best_phys, best_ld


def estimate_half_widths_from_sampler(sampler, prior_transform, fallback_half_widths):
    try:
        all_u = []
        all_ld = []
        for j in range(len(sampler.searched_points_list)):
            n = sampler.element_num_list[j]
            if n <= 0:
                continue
            all_u.append(sampler.searched_points_list[j][:n].copy())
            all_ld.append(sampler.searched_log_densities_list[j][:n].copy())
        if not all_u:
            return np.asarray(fallback_half_widths, dtype=float)

        u = np.concatenate(all_u, axis=0)
        ld = np.concatenate(all_ld, axis=0)
        top_k = min(max(200, u.shape[0] // 20), u.shape[0])
        idx = np.argsort(ld)[-top_k:]
        phys = prior_transform(u[idx])
        std = np.std(phys, axis=0)
        return np.maximum(3.0 * std, 0.35 * np.asarray(fallback_half_widths, dtype=float))
    except Exception:
        return np.asarray(fallback_half_widths, dtype=float)


def build_callback(stage, temper_state, schedule_history):
    ref = {'best': None, 'iter': 0}

    def callback(sampler, i):
        current = float(sampler.max_logden_list[0])

        if ref['best'] is None or current > ref['best']:
            ref['best'] = current
            ref['iter'] = i
            return

        current_stage = temper_state['stage']
        if i - ref['iter'] < stage.stop_stable_iters:
            if i % 1000 == 0 and i > 0:
                sampler.save_state()
            return

        if current_stage < len(stage.S_schedule) - 1:
            new_stage = current_stage + 1
            temper_state['stage'] = new_stage
            temper_state['S'] = stage.S_schedule[new_stage]
            ref['best'] = current
            ref['iter'] = i
            schedule_history.append({'iter': int(i), 'S': float(temper_state['S'])})
            print(
                f"[{stage.name}] Stuck for {stage.stop_stable_iters} iters, "
                f"raising S -> {temper_state['S']:.1f} at iter {i}",
                flush=True,
            )
        if i % 1000 == 0 and i > 0:
            sampler.save_state()

    return callback


def stage_summary_path(stage_dir):
    return os.path.join(stage_dir, 'stage_summary.json')


def run_stage(stage, seed_phys, prev_half_widths, output_root):
    stage_dir = os.path.join(output_root, stage.name)
    os.makedirs(stage_dir, exist_ok=True)

    loglike_obj, data_snr = make_loglike(stage)

    if stage.use_precomputed_seed:
        seed_phys, precomputed_info = maybe_precomputed_seed(loglike_obj, seed_phys)
    else:
        precomputed_info = None

    lo, hi, half_widths, seed_phys = make_bounds(seed_phys, prev_half_widths, stage)
    prior_transform, inverse_prior_transform = make_prior_functions(lo, hi)
    seed_u = inverse_prior_transform(seed_phys.reshape(1, -1))[0]

    temper_state = {'stage': 0, 'S': float(stage.S_schedule[0])}
    log_density = make_log_density(loglike_obj, temper_state)

    seed_ld = float(log_density(seed_phys.reshape(1, -1))[0])
    init_cov = np.eye(5, dtype=float) * (stage.init_u_sigma ** 2)

    config = parismc.SamplerConfig(
        merge_confidence=0.9,
        alpha=stage.alpha,
        trail_size=stage.trail_size,
        boundary_limiting=True,
        use_beta=True,
        integral_num=int(1e5),
        gamma=stage.gamma,
        exclude_scale_z=np.inf,
        use_pool=False,
        keep_dead_processes=True,
    )

    sampler = parismc.Sampler(
        ndim=5,
        n_seed=1,
        log_density_func=log_density,
        init_cov_list=[init_cov],
        prior_transform=prior_transform,
        config=config,
    )

    schedule_history = [{'iter': 0, 'S': float(stage.S_schedule[0])}]
    callback = build_callback(stage, temper_state, schedule_history)

    os.chdir(SEARCH_DIR)
    sampler.run_sampling(
        num_iterations=stage.num_iterations,
        savepath=stage_dir,
        print_iter=100,
        callback=callback,
        external_lhs_points=seed_u.reshape(1, -1),
        external_lhs_log_densities=np.array([seed_ld], dtype=float),
        stop_max_ld_stable_iters=stage.stop_stable_iters * 2,
    )

    best_phys, best_ld_tempered = get_best_point_and_logden(sampler, prior_transform)
    best_ld_raw = evaluate_one(loglike_obj, best_phys)
    next_half_widths = estimate_half_widths_from_sampler(sampler, prior_transform, half_widths)

    summary = {
        'stage': stage.name,
        'objective': stage.objective,
        'T_years': stage.T,
        'data_snr': data_snr,
        'bounds_lo': lo.tolist(),
        'bounds_hi': hi.tolist(),
        'seed_phys': seed_phys.tolist(),
        'seed_logden_tempered': seed_ld,
        'best_phys': best_phys.tolist(),
        'best_logden_raw': best_ld_raw,
        'best_logden_tempered': best_ld_tempered,
        'schedule_history': schedule_history,
        'next_half_widths': next_half_widths.tolist(),
        'precomputed_seed_info': precomputed_info,
    }
    with open(stage_summary_path(stage_dir), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[{stage.name}] seed: {seed_phys}")
    print(f"[{stage.name}] best raw loglike: {best_ld_raw:.6f}")
    print(f"[{stage.name}] best phys: {best_phys}")
    return best_phys, next_half_widths, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Staged PARIS continuation for EMRI intrinsic search.")
    parser.add_argument(
        '--output-root',
        default=DEFAULT_OUTPUT_ROOT,
        help='Directory for per-stage outputs.',
    )
    parser.add_argument(
        '--start-stage',
        choices=[stage.name for stage in DEFAULT_STAGES],
        default=DEFAULT_STAGES[0].name,
        help='Stage name to start from.',
    )
    parser.add_argument(
        '--seed-logm1', type=float, default=float(PARIS2_BEST_FIT[0]),
        help='Starting log10(m1) seed.'
    )
    parser.add_argument(
        '--seed-logm2', type=float, default=float(PARIS2_BEST_FIT[1]),
        help='Starting log10(m2) seed.'
    )
    parser.add_argument(
        '--seed-a', type=float, default=float(PARIS2_BEST_FIT[2]),
        help='Starting spin seed.'
    )
    parser.add_argument(
        '--seed-p0', type=float, default=float(PARIS2_BEST_FIT[3]),
        help='Starting p0 seed.'
    )
    parser.add_argument(
        '--seed-e0', type=float, default=float(PARIS2_BEST_FIT[4]),
        help='Starting e0 seed.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    start_idx = next(i for i, stage in enumerate(DEFAULT_STAGES) if stage.name == args.start_stage)
    stages = DEFAULT_STAGES[start_idx:]

    seed_phys = np.array(
        [args.seed_logm1, args.seed_logm2, args.seed_a, args.seed_p0, args.seed_e0],
        dtype=float,
    )
    prev_half_widths = None
    all_summaries = []

    for stage in stages:
        print(f"\n=== Running stage: {stage.name} ({stage.objective}, T={stage.T:.2f} yr) ===")
        seed_phys, prev_half_widths, summary = run_stage(stage, seed_phys, prev_half_widths, args.output_root)
        all_summaries.append(summary)

    final_summary_path = os.path.join(args.output_root, 'run_summary.json')
    with open(final_summary_path, 'w') as f:
        json.dump(all_summaries, f, indent=2)

    print("\nFinal staged result:")
    print(seed_phys)
    print(f"Saved run summary to {final_summary_path}")


if __name__ == '__main__':
    main()
