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
        if self.verbose:
            print(f"rho_tot={rho_tot:.6g}, rho_dom_M={rho_dom_M:.6g}, beta={beta:.6g}")
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
            print(f"{'group':>20} {'X_m':>12} {'rho_m':>12} {'(X_m-rho_m)^2':>16}")
            for idx, group in enumerate(selected):
                X_m_i = float(X_modes[idx])
                rho_m_i = float(rho_m[idx])
                contrib = (X_m_i - rho_m_i) ** 2
                print(f"{str(group):>20} {X_m_i:>12.6g} {rho_m_i:>12.6g} {contrib:>16.6g}")
            print(f"X_scalar={X_scalar:.6g}, chi_sq={chi_sq:.6g}, f_stat={f_stat:.6g}")
            print(f"Pure log-likelihood: {f_stat:.6g}")

        return float(f_stat)

    def log_density_batch(self, theta_batch):
        """
        Batched pure (non-maximized) f-statistic for B templates at once.

        Waveform generation is still one call per row — the underlying `few`
        waveform generator (and hence ResponseWrapper) only accepts scalar
        source parameters, so there is no batched API to call into there.
        What's batched is everything downstream of generation: the per-row
        waveforms are stacked into (B, n_chan, N) arrays and every rfft /
        inner-product / chi-square step runs once across the whole batch
        (via GWfuncs_noise.freq_wave_batch / inner_batch) instead of once
        per row.

        Parameters
        ----------
        theta_batch : (B, 14) array of physical template parameters
            [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
             Phi_phi0, Phi_theta0, Phi_r0]

        Returns
        -------
        (B,) numpy array of f_stat values (-inf where waveform generation
        raised or beta <= 0, mirroring __call__'s behavior). Otherwise can
        be negative, same as __call__.
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

        signal_fft_arr = self.signal_fft  # (n_chan, N_freq), from freq_wave

        h_temp_batch = xp.stack(h_temp_list, axis=0)  # (B, n_chan, N)
        h_temp_fft_batch = self.gwf.freq_wave_batch(h_temp_batch)

        rho_tot = xp.sqrt(self.gwf.inner_batch(h_temp_fft_batch, h_temp_fft_batch))
        X_scalar = self.gwf.inner_batch(signal_fft_arr, h_temp_fft_batch) / rho_tot

        rho_m = xp.empty((B, n_modes), dtype=xp.float64)
        X_modes = xp.empty((B, n_modes), dtype=xp.float64)
        for m in range(n_modes):
            hf_batch = self.gwf.freq_wave_batch(xp.stack(mode_wf_lists[m], axis=0))
            rho_m[:, m] = xp.sqrt(self.gwf.inner_batch(hf_batch, hf_batch))

        max_rho_idx = xp.argmax(rho_m, axis=-1)
        idx = xp.arange(B)
        rho_dom_M = rho_m[idx, max_rho_idx]
        beta = self.gwf.calc_beta(rho_dom_M, rho_tot)

        chi_sq = xp.sum((X_modes - rho_m) ** 2, axis=-1)
        f_stat = X_scalar * xp.exp(-0.5 * beta * chi_sq)
        if self.verbose:
            print(f" X_scalar: {X_scalar}")
            print(f" chi_sq: {chi_sq}")
            print(f" beta: {beta}")
            # print(f"Batch f_stat (before masking): {f_stat}")

        f_stat = xp.where(beta <= 0.0, -xp.inf, f_stat)
        f_stat = xp.where(xp.asarray(ok), f_stat, -xp.inf)
        if self.verbose:
            print(f" log-likelihood: {f_stat}")

        f_stat_np = f_stat.get() if hasattr(f_stat, 'get') else f_stat
        return np.asarray(f_stat_np, dtype=np.float64)
