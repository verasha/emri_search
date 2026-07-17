import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI


class LogLike:
    """
    F-statistic log-likelihood with time-maximization applied only to X_scalar.

    rho_tot, rho_m, and X_modes all use the standard (zero-lag) inner product.
    Only X_scalar = max_τ |<d|h(τ)>| / rho_h uses time-maximization.

    This isolates the effect of time-maximizing the main detection statistic
    without affecting the chi-square veto normalization.
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
                 n_vals=None,
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

        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params
        self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, dist,
                                   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,
                                   T=self.T, dt=self.dt)

        if self.mode_select:
            self.selected_labels = self.mode_select
        else:
            mode_selector = ModeSelector(self.params, self.traj, self.amp,
                                         self.ylm_gen, self.delta_T, self.gwf,
                                         verbose=self.verbose)
            self.selected_modes, self.selected_labels = mode_selector.select_modes(
                ell=ell, n_vals=n_vals, M_sel=M_mode)
            self.flattened_modes = []
            for group in self.selected_labels:
                self.flattened_modes.extend(group)
            if self.verbose:
                print(f"Selected modes: {self.selected_labels}")

    def _generate_selected_waveforms(self, params, selected_labels):
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params
        waveforms_per_group = []
        for group in selected_labels:
            wf = self.waveform_gen(
                m1, m2, a, p0, e0, xI0, dist,
                qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,
                dt=self.dt, T=self.T,
                mode_selection=group,
                include_minus_mkn=False,
            )
            waveforms_per_group.append(wf)
        return self.gwf.xp.stack(waveforms_per_group, axis=1)

    def __call__(self, theta_template):
        selected = self.mode_select if self.mode_select else self.selected_labels
        waveform_combined = self._generate_selected_waveforms(theta_template, selected)

        m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t = theta_template
        h_temp = self.waveform_gen(
            m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t,
            qS_t, phiS_t, qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
            dt=self.dt, T=self.T,
        )

        # Standard (zero-lag) norms
        rho_tot = self.gwf.rhostat(h_temp)

        waveform_per_mode = [waveform_combined[:, i] for i in range(waveform_combined.shape[1])]

        rho_m = self.gwf.xp.empty(len(waveform_per_mode), dtype=self.gwf.xp.float64)
        for idx, wf in enumerate(waveform_per_mode):
            rho_m[idx] = self.gwf.rhostat(wf)

        max_rho_idx = rho_m.argmax()
        rho_dom_M = rho_m[max_rho_idx]

        # TIME-MAXIMIZED X_scalar only
        X_scalar = self.gwf.Xstat_timemax(self.signal, h_temp)

        # Standard X_modes (no timemax)
        X_modes = self.gwf.Xmstat(self.signal, waveform_per_mode, rho_m)

        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)
        if float(beta) <= 0.0:
            return -np.inf

        if self.verbose:
            print(f"rho_tot={rho_tot:.4g}, rho_dom_M={rho_dom_M:.4g}, beta={beta:.4g}")
            print(f"X_scalar (timemax): {X_scalar:.4g}")

        chi_sq = self.gwf.chi_sq(X_modes, rho_m)
        f_stat = X_scalar * self.gwf.xp.exp(-0.5 * beta * chi_sq)

        f_stat_real = self.gwf.xp.real(f_stat)
        logl_res = float(f_stat_real.get() if hasattr(f_stat_real, 'get') else f_stat_real)

        if self.verbose:
            print(f"Log-likelihood (timemax X only): {logl_res:.6g}")

        return logl_res
