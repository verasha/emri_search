import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import YRSID_SI


class LogLike:
    """
    Hybrid time-max / standard log-likelihood with TDI AET channels and optional noise.

    Time maximization is applied ONLY to X_scalar (the full cross-correlation peak).
    rho_tot, rho_m, and X_modes all use the standard noise-weighted inner product
    (same as phasemax, evaluated at tau=0).

    Compared to full timemax:
    - X_scalar still finds tau* from the full-template cross-correlation
    - X_modes are NOT evaluated at tau* — they use |<d|h_m>| at tau=0
    - chi_sq is therefore more selective: noise that inflated X_scalar at tau*
      does not also inflate X_modes, so wrong-param chi_sq suppression is stronger
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

        # wave_fft for timemax cross-correlation (full FFT format)
        self.signal_fft = gwf.wave_fft(self.signal)
        # freq_wave for standard inner product (rfft format)
        self.signal_fft_r = gwf.freq_wave(self.signal)

        self.delta_T = self.T * YRSID_SI / self.N_traj
        if self.mode_select:
            if self.verbose:
                print(f"Using externally provided modes: {self.mode_select}")
            self.selected_labels = self.mode_select
        else:
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
        Evaluate hybrid tmXonly log-likelihood.

        X_scalar = max_tau |<d|h(tau)>| / rho_h  (time-maximized)
        rho_tot, rho_m = sqrt(<h|h>)             (standard inner product)
        X_modes = |<d|h_m>| / rho_m              (standard inner product, tau=0)

        Returns float: f-statistic value.
        """
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = theta_template

        selected = self.mode_select if self.mode_select else self.selected_labels
        waveform_combined = self._generate_selected_waveforms(theta_template, selected)

        h_temp = self.gwf.xp.array(self.waveform_response(
            m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
            T=self.T, dt=self.dt,
        ))

        # Standard rho computations (rfft inner product, no time max)
        h_temp_fft_r = self.gwf.freq_wave(h_temp)
        rho_tot = self.gwf.xp.sqrt(self.gwf.inner(h_temp_fft_r, h_temp_fft_r))

        waveform_per_mode = [waveform_combined[i] for i in range(waveform_combined.shape[0])]
        mode_ffts_r = [self.gwf.freq_wave(wf) for wf in waveform_per_mode]

        rho_m = self.gwf.xp.empty(len(mode_ffts_r), dtype=self.gwf.xp.float64)
        for idx, hf_r in enumerate(mode_ffts_r):
            rho_m[idx] = self.gwf.xp.sqrt(self.gwf.inner(hf_r, hf_r))

        rho_dom_M = rho_m[rho_m.argmax()]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)
        if float(beta) <= 0.0:
            return -np.inf

        # X_scalar: time-maximized cross-correlation (full FFT format)
        h_temp_fft = self.gwf.wave_fft(h_temp)
        S_full = self.gwf.cross_corr_f(self.signal_fft, h_temp_fft)
        tau_star = int(self.gwf.xp.argmax(self.gwf.xp.abs(S_full)))
        X_scalar = float(self.gwf.xp.abs(S_full[tau_star])) / float(rho_tot)

        # X_modes: standard inner product at tau=0 (NOT at tau*)
        X_modes = self.gwf.xp.empty(len(mode_ffts_r), dtype=self.gwf.xp.float64)
        for idx, hf_r in enumerate(mode_ffts_r):
            inner_complex = self.gwf.inner(self.signal_fft_r, hf_r, return_complex=True)
            X_modes[idx] = float(self.gwf.xp.abs(inner_complex)) / float(rho_m[idx])

        chi_sq = self.gwf.chi_sq(X_modes, rho_m)
        f_stat = X_scalar * float(self.gwf.xp.exp(-0.5 * beta * chi_sq))

        if self.verbose:
            print(f"tmXonly Log-likelihood: {f_stat:.6g}  "
                  f"(X_scalar={X_scalar:.4g}, tau*={tau_star}, chi_sq={float(chi_sq):.4g})")

        return float(f_stat)
