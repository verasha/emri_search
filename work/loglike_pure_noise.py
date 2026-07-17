import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import YRSID_SI


class LogLike:
    """
    Pure (non-maximized) log-likelihood with TDI AET channels and optional noise.

    Uses Re(<d|h>) / rho_h for X_scalar — no time shift, no phase rotation.
    X_scalar can be negative at wrong params, so the noise floor is centered at 0
    rather than the positive Rayleigh bias of phasemax.
    """

    def __init__(self, params,
                 waveform_response,
                 gwf,
                 add_noise=False,
                 seed=0,
                 M_mode=5,
                 N_traj=5000,
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

        self.signal_fft = gwf.freq_wave(self.signal)

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
        Evaluate pure (non-maximized) log-likelihood.

        X_scalar = Re(<d|h>) / rho_h — no abs, no phase rotation, no time shift.
        Can return negative values at wrong params (noise anti-correlation).

        Returns float: f-statistic value (can be negative).
        """
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = theta_template

        selected = self.mode_select if self.mode_select else self.selected_labels
        waveform_combined = self._generate_selected_waveforms(theta_template, selected)

        h_temp = self.gwf.xp.array(self.waveform_response(
            m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
            T=self.T, dt=self.dt,
        ))
        h_temp_fft = self.gwf.freq_wave(h_temp)

        waveform_per_mode = [waveform_combined[i] for i in range(waveform_combined.shape[0])]
        mode_ffts = [self.gwf.freq_wave(wf) for wf in waveform_per_mode]

        rho_tot = self.gwf.xp.sqrt(self.gwf.inner(h_temp_fft, h_temp_fft))

        rho_m = self.gwf.xp.empty(len(mode_ffts), dtype=self.gwf.xp.float64)
        for idx, hf in enumerate(mode_ffts):
            rho_m[idx] = self.gwf.xp.sqrt(self.gwf.inner(hf, hf))

        rho_dom_M = rho_m[rho_m.argmax()]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)
        if float(beta) <= 0.0:
            return -np.inf

        # Pure: Re(<d|h>) / rho_h — no abs, no phase factor
        X_scalar = float(self.gwf.inner(self.signal_fft, h_temp_fft, return_complex=False)) / float(rho_tot)

        # Per-mode: Re(<d|h_m>) / rho_m — no phase projection
        X_modes = self.gwf.xp.empty(len(mode_ffts), dtype=self.gwf.xp.float64)
        for idx, hf in enumerate(mode_ffts):
            X_modes[idx] = float(self.gwf.inner(self.signal_fft, hf, return_complex=False)) / float(rho_m[idx])

        chi_sq = self.gwf.chi_sq(X_modes, rho_m)
        f_stat = X_scalar * float(self.gwf.xp.exp(-0.5 * beta * chi_sq))

        if self.verbose:
            print(f"Pure log-likelihood: {f_stat:.6g}")

        return float(f_stat)
