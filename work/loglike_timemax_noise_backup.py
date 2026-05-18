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
                 add_kwargs=None,
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
        - add_kwargs: dict with chi2, evolve_1PA, evolve_primary, evolve_2PA,
                      deviation_included, dev_1, dev_2
        - add_noise: whether to add colored noise to the signal
        - seed: random seed for noise generation
        - mode_select: pre-selected modes [(l,m,n)]; if None, runs ModeSelector
        """
        self.params = params
        self.waveform_response = waveform_response
        self.gwf = gwf
        self.dt = gwf.dt
        self.T = gwf.T
        self.add_kwargs = add_kwargs or {}
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

        # Build positional + keyword args matching inference.py convention
        wave_params = [
            m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
            Phi_phi0, Phi_theta0, Phi_r0,
            self.add_kwargs.get('chi2'), self.add_kwargs.get('evolve_1PA'),
            self.add_kwargs.get('evolve_primary'), self.add_kwargs.get('evolve_2PA'),
            self.add_kwargs.get('deviation_included'), self.add_kwargs.get('dev_1'),
            self.add_kwargs.get('dev_2'),
        ]
        emri_kwargs = {"T": self.T, "dt": self.dt, **self.add_kwargs}

        # Generate true signal — shape (3, N), real AET
        self.signal = gwf.xp.array(waveform_response(*wave_params, **emri_kwargs))

        # Optionally add colored noise (time domain, shape (3, N))
        if add_noise:
            noise = gwf.generate_colored_noise(seed=seed)
            self.signal = self.signal + noise
            if verbose:
                print(f"[INFO] Added colored noise with seed={seed}")

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
            wave_params = [
                m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                self.add_kwargs.get('chi2'), self.add_kwargs.get('evolve_1PA'),
                self.add_kwargs.get('evolve_primary'), self.add_kwargs.get('evolve_2PA'),
                self.add_kwargs.get('deviation_included'), self.add_kwargs.get('dev_1'),
                self.add_kwargs.get('dev_2'),
            ]
            emri_kwargs = {
                "T": self.T, "dt": self.dt,
                **self.add_kwargs,
                "mode_selection": group,
                "include_minus_mkn": False,
            }
            wf = self.gwf.xp.array(self.waveform_response(*wave_params, **emri_kwargs))
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

        wave_params = [
            m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t,
            Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
            self.add_kwargs.get('chi2'), self.add_kwargs.get('evolve_1PA'),
            self.add_kwargs.get('evolve_primary'), self.add_kwargs.get('evolve_2PA'),
            self.add_kwargs.get('deviation_included'), self.add_kwargs.get('dev_1'),
            self.add_kwargs.get('dev_2'),
        ]
        emri_kwargs = {"T": self.T, "dt": self.dt, **self.add_kwargs}

        # Generate full template — shape (3, N)
        h_temp = self.gwf.xp.array(self.waveform_response(*wave_params, **emri_kwargs))

        rho_tot = self.gwf.rhostat_timemax(h_temp)

        # Split combined waveform into per-mode (3, N) arrays
        waveform_per_mode = [waveform_combined[i] for i in range(waveform_combined.shape[0])]

        rho_m = self.gwf.xp.empty(len(waveform_per_mode), dtype=self.gwf.xp.float64)
        for idx, wf in enumerate(waveform_per_mode):
            rho_m[idx] = self.gwf.rhostat_timemax(wf)

        max_rho_idx = rho_m.argmax()

        if self.verbose:
            print(f"Waveform amplitudes (time-maximized):")
            for i, wf in enumerate(waveform_per_mode):
                ip = self.gwf.inner_timemax(wf, wf)
                print(f"  Mode {i}: <hf|hf>_timemax={ip:.4g}, rho={rho_m[i]:.4g}")
            print(f"rho_m: {rho_m}")
            print(f"Dominant mode index: {max_rho_idx}, rho: {rho_m[max_rho_idx]:.4g}")

        X_modes = self.gwf.Xmstat_timemax(self.signal, waveform_per_mode, rho_m)
        X_scalar = self.gwf.Xstat_timemax(self.signal, h_temp)

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
