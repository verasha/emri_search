import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import YRSID_SI


class LogLike:
    """
    Time-marginalized log-likelihood with TDI AET channels and optional noise.

    Replaces time-maximization (argmax_τ |<d|h(τ)>|) with time-marginalization
    (mean_τ F(τ)). This averages the F-statistic over all time lags rather than
    picking the single noisiest peak, making it more robust to false noise modes:
      max_τ |noise cross-corr| ~ σ√(2 log N) ≈ 5.3σ  (N=1.5M, 3 months @ 5s)
      mean_τ |noise cross-corr| ~ σ√(π/2)    ≈ 1.25σ
    """

    def __init__(self, params,
                 waveform_response,
                 gwf,
                 add_noise=False,
                 seed=0,
                 M_mode=5,
                 N_traj=5000,
                 mode_threshold=0.01,
                 verbose=False,
                 mode_select=None,
                 ell=2,
                 n_vals=None,
                 ):
        self.params = params
        self.waveform_response = waveform_response
        self.gwf = gwf
        self.dt = gwf.dt
        self.T = gwf.T
        self.M_mode = M_mode
        self.N_traj = N_traj
        self.mode_threshold = mode_threshold
        self.verbose = verbose
        self.mode_select = mode_select

        inner_gen = waveform_response.waveform_gen.waveform_generator
        self.traj = getattr(inner_gen, 'inspiral_generator', None)
        self.amp = getattr(inner_gen, 'amplitude_generator', None)
        self.interpolate_mode_sum = getattr(inner_gen, 'create_waveform', None)
        self.ylm_gen = getattr(inner_gen, 'ylm_gen', None)

        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        self.signal = gwf.xp.array(waveform_response(
            m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
            T=self.T, dt=self.dt,
        ))

        if add_noise:
            noise = gwf.generate_colored_noise(seed=seed)
            self.signal = self.signal + noise
            if verbose:
                print(f"[INFO] Added colored noise with seed={seed}")

        # wave_fft (full FFT, length N) — required for cross_corr_f / F_stat_timemarg_f
        self.signal_fft = gwf.wave_fft(self.signal)

        self.delta_T = self.T * YRSID_SI / self.N_traj
        if self.mode_select:
            if self.verbose:
                print(f"Using externally provided modes: {self.mode_select}")
            self.selected_labels = self.mode_select
        else:
            if self.verbose:
                print(f"Delta_T for mode selection: {self.delta_T} seconds")
            mode_selector = ModeSelector(self.params, self.traj, self.amp,
                                         self.ylm_gen, self.delta_T, self.gwf,
                                         verbose=self.verbose)
            self.selected_modes, self.selected_labels = mode_selector.select_modes(
                ell=ell,
                n_vals=n_vals,
                M_sel=M_mode,
            )
            self.flattened_modes = []
            for group in self.selected_labels:
                self.flattened_modes.extend(group)

            if self.verbose:
                print(f"Selected modes: {self.selected_labels}")
                print(f"Flattened modes: {self.flattened_modes}")

    def _generate_selected_waveforms(self, params, selected_labels):
        """Generate per-mode-group waveforms. Returns (N_groups, 3, N) array."""
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        waveforms_per_group = []
        for group in selected_labels:
            wf = self.gwf.xp.array(self.waveform_response(
                m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                T=self.T, dt=self.dt,
                mode_selection=group,
                include_minus_mkn=False,
            ))
            waveforms_per_group.append(wf)

        return self.gwf.xp.stack(waveforms_per_group, axis=0)

    def __call__(self, theta_template):
        """
        Evaluate time-marginalized log-likelihood.

        Returns float: mean_τ F(τ)  (always ≥ 0 since |·| and exp(·) are non-negative)
        """
        selected = self.mode_select if self.mode_select else self.selected_labels
        waveform_combined = self._generate_selected_waveforms(theta_template, selected)

        m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t, \
            Phi_phi0_t, Phi_theta0_t, Phi_r0_t = theta_template

        h_temp = self.gwf.xp.array(self.waveform_response(
            m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t,
            Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
            T=self.T, dt=self.dt,
        ))

        h_temp_fft = self.gwf.wave_fft(h_temp)
        waveform_per_mode = [waveform_combined[i] for i in range(waveform_combined.shape[0])]
        mode_ffts = [self.gwf.wave_fft(wf) for wf in waveform_per_mode]

        # rho_tot and rho_m: template SNRs (time-max autocorrelation ≈ standard <h|h>)
        rho_tot = self.gwf.xp.sqrt(self.gwf.inner_timemax_f(h_temp_fft, h_temp_fft))

        rho_m = self.gwf.xp.empty(len(mode_ffts), dtype=self.gwf.xp.float64)
        for idx, hf in enumerate(mode_ffts):
            rho_m[idx] = self.gwf.xp.sqrt(self.gwf.inner_timemax_f(hf, hf))

        max_rho_idx = rho_m.argmax()
        rho_dom_M = rho_m[max_rho_idx]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        if float(beta) <= 0.0:
            return -np.inf

        if self.verbose:
            print(f"beta={beta:.4g}  rho_dom_M={rho_dom_M:.4g}  rho_tot={rho_tot:.4g}")

        # Time-marginalized F-statistic: mean_τ F(τ)
        f_marg = self.gwf.F_stat_timemarg_f(
            self.signal_fft, h_temp_fft, mode_ffts,
            rho_tot, rho_m, beta,
        )

        if self.verbose:
            print(f"Time-marginalized log-likelihood: {f_marg:.6g}")

        return f_marg
