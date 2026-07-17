import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI

class LogLikeTimeOnly:
    """
    Time-maximized (no phase-maximized) log-likelihood.

    X_scalar = max_τ Re(<x|h(τ)>) / sqrt(<h|h>)

    Maximizes over time shifts but uses Re() (not abs()), so phase is NOT marginalized.
    Compare with:
      - loglike_pure:     Re(<x|h>)        fixed τ, fixed phase
      - loglike_phase:    |<x|h>|          fixed τ, phase-max
      - loglike_timeonly: max_τ Re(<x|h>)  time-max only        ← this file
      - loglike_timemax:  max_τ |<x|h>|    time-max + phase-max
    """

    def __init__(self, params,
                 waveform_gen,
                 gwf,
                 M_init=100,
                 M_mode=5,
                 N_traj=5000,
                 mode_threshold=0.01,
                 verbose=False,
                 waveform_gen_sep=None,
                 noise_weighted=False,
                 mode_select=None,
                 ell=2,
                 n_vals=None
                 ):
        self.params = params
        self.waveform_gen = waveform_gen
        self.waveform_gen_sep = waveform_gen_sep if waveform_gen_sep is not None else waveform_gen
        self.M_init = M_init
        self.M_mode = M_mode
        self.N_traj = N_traj
        self.mode_threshold = mode_threshold
        self.verbose = verbose
        self.mode_select = mode_select

        self.gwf = gwf
        self.dt = gwf.dt
        self.T = gwf.T

        self.traj = getattr(waveform_gen.waveform_generator, 'inspiral_generator', None)
        self.amp = getattr(waveform_gen.waveform_generator, 'amplitude_generator', None)
        self.interpolate_mode_sum = getattr(waveform_gen.waveform_generator, 'create_waveform', None)
        self.ylm_gen = getattr(waveform_gen.waveform_generator, 'ylm_gen', None)

        self.delta_T = self.T * YRSID_SI / self.N_traj
        if self.mode_select:
            if self.verbose:
                print(f"Using externally provided modes: {self.mode_select}")
            self.selected_labels = self.mode_select
        if not self.mode_select:
            if self.verbose:
                print(f"Delta_T for mode selection: {self.delta_T} seconds")

        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params
        self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, dist,
                                   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,
                                   T=self.T, dt=self.dt)

        if not self.mode_select:
            if self.verbose:
                print("Generating modes at initialization...")
            mode_selector = ModeSelector(self.params, self.traj, self.amp,
                                         self.ylm_gen, self.delta_T, self.gwf,
                                         verbose=self.verbose)
            self.selected_modes, self.selected_labels = mode_selector.select_modes(
                ell=ell, n_vals=n_vals, M_sel=M_mode
            )
            self.flattened_modes = []
            for group in self.selected_labels:
                self.flattened_modes.extend(group)
            if self.verbose:
                print(f"Selected modes: {self.selected_labels}")

    def _generate_selected_waveforms(self, params, selected_labels):
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params
        waveforms_per_group = []
        for group in self.selected_labels:
            waveform_group = self.waveform_gen(
                m1, m2, a, p0, e0, xI0, dist,
                qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                dt=self.dt, T=self.T,
                mode_selection=group,
                include_minus_mkn=False,
            )
            waveforms_per_group.append(waveform_group)
        return self.gwf.xp.stack(waveforms_per_group, axis=1)

    def __call__(self, theta_template):
        """
        Time-maximized (no phasemax) log-likelihood.
        X_scalar = max_τ Re(<x|h(τ)>) / sqrt(<h|h>)
        """
        if self.mode_select:
            waveform_combined = self._generate_selected_waveforms(theta_template, self.mode_select)
        else:
            waveform_combined = self._generate_selected_waveforms(theta_template, self.flattened_modes)

        m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t = theta_template

        h_temp = self.waveform_gen(
            m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t,
            qS_t, phiS_t, qK_t, phiK_t,
            Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
            dt=self.dt, T=self.T
        )

        rho_tot = self.gwf.rhostat(h_temp)

        waveform_per_mode = [waveform_combined[:, i] for i in range(waveform_combined.shape[1])]

        rho_m = self.gwf.xp.empty(len(waveform_per_mode), dtype=self.gwf.xp.float64)
        for idx, wf in enumerate(waveform_per_mode):
            rho_m[idx] = self.gwf.rhostat(wf)

        max_rho_idx = rho_m.argmax()

        # TIME-ONLY: mirrors loglike_timemax but Re() instead of abs() per mode
        X_scalar = self.gwf.Xstat_timeonly(self.signal, h_temp)
        X_modes = self.gwf.Xmstat_timeonly(self.signal, waveform_per_mode, rho_m)

        rho_dom_M = rho_m[max_rho_idx]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        if self.verbose:
            print(f"rho_dom_M: {rho_dom_M}, rho_tot: {rho_tot}, beta: {beta}")
            print(f"X_scalar (time-only): {X_scalar}")

        chi_sq = self.gwf.chi_sq(X_modes, rho_m)
        f_stat = X_scalar * self.gwf.xp.exp(-0.5 * beta * chi_sq)

        f_stat_real = self.gwf.xp.real(f_stat)
        logl_res = float(f_stat_real.get()) if hasattr(f_stat_real, 'get') else float(f_stat_real)

        if self.verbose:
            print(f"Log-likelihood (time-only): {logl_res}")

        return logl_res
