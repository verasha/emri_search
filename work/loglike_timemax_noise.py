import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import YRSID_SI


class LogLike:
    """
    Time-maximized log-likelihood with TDI AET channels and optional noise.

    Replaces waveform_gen with a ResponseWrapper (waveform_response) that returns
    real (3, N) AET arrays. Uses GravWaveAnalysis from GWfuncs_noise.py.
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
        """
        Parameters:
        - params: [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
        - waveform_response: ResponseWrapper instance (returns (3, N) real AET)
        - gwf: GravWaveAnalysis instance from GWfuncs_noise.py
        - add_noise: whether to add colored noise to the signal
        - seed: random seed for noise generation
        - mode_select: pre-selected modes [(l,m,n)]; if None, runs ModeSelector
        """
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

        # Extract trajectory/amplitude/ylm from ResponseWrapper internals for ModeSelector
        inner_gen = waveform_response.waveform_gen.waveform_generator
        self.traj = getattr(inner_gen, 'inspiral_generator', None)
        self.amp = getattr(inner_gen, 'amplitude_generator', None)
        self.interpolate_mode_sum = getattr(inner_gen, 'create_waveform', None)
        self.ylm_gen = getattr(inner_gen, 'ylm_gen', None)

        # Unpack signal parameters
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        emri_kwargs = {"T": self.T, "dt": self.dt}

        # Generate true signal — shape (3, N), real AET
        self.signal = gwf.xp.array(waveform_response(
            m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
            **emri_kwargs,
        ))

        # Optionally add colored noise (time domain, shape (n_chan, N))
        if add_noise:
            noise = gwf.generate_colored_noise(seed=seed)
            self.signal = self.signal + noise
            if verbose:
                print(f"[INFO] Added colored noise with seed={seed}")

        # Pre-compute signal FFTs once — reused across all __call__ evaluations
        self.signal_fft = gwf.wave_fft(self.signal)

        # Mode selection
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
        """
        Generate per-mode-group waveforms via ResponseWrapper.

        Returns array of shape (N_groups, 3, N) — axis 0 is group index.
        """
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

        # shape: (N_groups, 3, N)
        return self.gwf.xp.stack(waveforms_per_group, axis=0)

    def __call__(self, theta_template):
        """
        Evaluate time-maximized log-likelihood for template parameters.

        Returns float: time-maximized f-statistic log-likelihood.
        """
        selected = self.mode_select if self.mode_select else self.selected_labels
        waveform_combined = self._generate_selected_waveforms(theta_template, selected)
        # waveform_combined: (N_groups, 3, N)

        m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t = theta_template

        # Generate full template — shape (n_chan, N)
        h_temp = self.gwf.xp.array(self.waveform_response(
            m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t,
            Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
            T=self.T, dt=self.dt,
        ))

        # Pre-compute FFTs once — reused for rho and X computations
        h_temp_fft = self.gwf.wave_fft(h_temp)
        waveform_per_mode = [waveform_combined[i] for i in range(waveform_combined.shape[0])]
        mode_ffts = [self.gwf.wave_fft(wf) for wf in waveform_per_mode]

        rho_tot = self.gwf.xp.sqrt(self.gwf.inner_timemax_f(h_temp_fft, h_temp_fft))

        rho_m = self.gwf.xp.empty(len(mode_ffts), dtype=self.gwf.xp.float64)
        for idx, hf in enumerate(mode_ffts):
            rho_m[idx] = self.gwf.xp.sqrt(self.gwf.inner_timemax_f(hf, hf))

        max_rho_idx = rho_m.argmax()

        if self.verbose:
            print(f"Waveform amplitudes (time-maximized):")
            for i, hf in enumerate(mode_ffts):
                print(f"  Mode {i}: <hf|hf>_timemax={rho_m[i]**2:.4g}, rho={rho_m[i]:.4g}")
            print(f"rho_m: {rho_m}")
            print(f"Dominant mode index: {max_rho_idx}, rho: {rho_m[max_rho_idx]:.4g}")

        # Find τ* from full template, then evaluate all modes at the same τ*
        S_full = self.gwf.cross_corr_f(self.signal_fft, h_temp_fft)
        tau_star = int(self.gwf.xp.argmax(self.gwf.xp.abs(S_full)))
        X_scalar = float(self.gwf.xp.abs(S_full[tau_star])) / float(rho_tot)

        X_modes = self.gwf.xp.empty(len(mode_ffts), dtype=self.gwf.xp.float64)
        for idx, hf in enumerate(mode_ffts):
            S_mode = self.gwf.cross_corr_f(self.signal_fft, hf)
            X_modes[idx] = float(self.gwf.xp.abs(S_mode[tau_star])) / float(rho_m[idx])

        rho_dom_M = rho_m[max_rho_idx]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        if float(beta) <= 0.0:
            return -np.inf

        if self.verbose:
            print(f"beta={beta:.4g}, rho_dom_M={rho_dom_M:.4g}, rho_tot={rho_tot:.4g}")
            print(f"X_scalar (time-maximized): {X_scalar:.4g}")

        chi_sq = self.gwf.chi_sq(X_modes, rho_m)
        f_stat = X_scalar * self.gwf.xp.exp(-0.5 * beta * chi_sq)

        f_stat_real = self.gwf.xp.real(f_stat)
        logl_res = float(f_stat_real.get() if hasattr(f_stat_real, 'get') else f_stat_real)

        if self.verbose:
            print(f"Time-maximized Log-likelihood: {logl_res:.6g}")

        return logl_res

    def log_density_batch(self, theta_batch):
        """
        Batched time-maximized f-statistic for B templates at once.

        Waveform generation is still one call per row — the underlying `few`
        waveform generator (and hence ResponseWrapper) only accepts scalar
        source parameters, so there is no batched API to call into there.
        What's batched is everything downstream of generation: the per-row
        waveforms are stacked into (B, n_chan, N) arrays and every FFT /
        cross-correlation / rho / chi-square step runs once across the whole
        batch (via GWfuncs_noise.wave_fft_batch / cross_corr_batch /
        inner_timemax_batch) instead of once per row.

        Parameters
        ----------
        theta_batch : (B, 14) array of physical template parameters
            [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
             Phi_phi0, Phi_theta0, Phi_r0]

        Returns
        -------
        (B,) numpy array of f_stat values (-inf where waveform generation
        raised or beta <= 0, mirroring __call__'s behavior).
        """
        theta_batch = np.asarray(theta_batch)
        B = theta_batch.shape[0]
        xp = self.gwf.xp

        selected = self.mode_select if self.mode_select else self.selected_labels
        n_modes = len(selected)

        h_temp_list = []
        mode_wf_lists = [[] for _ in range(n_modes)]
        ok = np.ones(B, dtype=bool)

        for b in range(B):
            theta = theta_batch[b]
            (m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t,
             qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t) = theta
            try:
                h_temp = xp.array(self.waveform_response(
                    m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t,
                    qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
                    T=self.T, dt=self.dt,
                ))
                wf_groups = self._generate_selected_waveforms(theta, selected)
            except Exception:
                ok[b] = False
                h_temp_list.append(xp.zeros_like(self.signal))
                for m in range(n_modes):
                    mode_wf_lists[m].append(xp.zeros_like(self.signal))
                continue

            h_temp_list.append(h_temp)
            for m in range(n_modes):
                mode_wf_lists[m].append(wf_groups[m])

        if not np.any(ok):
            return np.full(B, -np.inf)

        signal_fft_arr = xp.stack(self.signal_fft, axis=0)  # (n_chan, N)

        h_temp_batch = xp.stack(h_temp_list, axis=0)  # (B, n_chan, N)
        h_temp_fft_batch = self.gwf.wave_fft_batch(h_temp_batch)

        rho_tot = xp.sqrt(self.gwf.inner_timemax_batch(h_temp_fft_batch, h_temp_fft_batch))

        S_full = self.gwf.cross_corr_batch(signal_fft_arr, h_temp_fft_batch)  # (B, N)
        tau_star = xp.argmax(xp.abs(S_full), axis=-1)  # (B,)
        idx = xp.arange(B)
        X_scalar = xp.abs(S_full[idx, tau_star]) / rho_tot

        rho_m = xp.empty((B, n_modes), dtype=xp.float64)
        X_modes = xp.empty((B, n_modes), dtype=xp.float64)
        for m in range(n_modes):
            hf_batch = self.gwf.wave_fft_batch(xp.stack(mode_wf_lists[m], axis=0))
            rho_m[:, m] = xp.sqrt(self.gwf.inner_timemax_batch(hf_batch, hf_batch))
            S_mode = self.gwf.cross_corr_batch(signal_fft_arr, hf_batch)
            X_modes[:, m] = xp.abs(S_mode[idx, tau_star]) / rho_m[:, m]

        max_rho_idx = xp.argmax(rho_m, axis=-1)
        rho_dom_M = rho_m[idx, max_rho_idx]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        chi_sq = xp.sum((X_modes - rho_m) ** 2, axis=-1)
        f_stat = xp.real(X_scalar * xp.exp(-0.5 * beta * chi_sq))

        f_stat = xp.where(beta <= 0.0, -xp.inf, f_stat)
        f_stat = xp.where(xp.asarray(ok), f_stat, -xp.inf)

        f_stat_np = f_stat.get() if hasattr(f_stat, 'get') else f_stat
        return np.asarray(f_stat_np, dtype=np.float64)
