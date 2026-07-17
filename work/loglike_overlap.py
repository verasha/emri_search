import numpy as np
from modeselectoralt import ModeSelector
from few.utils.constants import Gpc, MRSUN_SI, YRSID_SI


class LogLike:
    """
    Compute overlap only 
    """

    def __init__(self, params,
                 waveform_gen,
                 gwf,
                 N_traj=5000,
                 verbose=False,
                 waveform_gen_sep=None,
                 ):
        """
        Parameters:
        - params: [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
        - waveform_gen: waveform generator built at T_max
        - gwf: GravWaveAnalysis built at T_max
        """
        self.params = params
        self.waveform_gen = waveform_gen
        self.waveform_gen_sep = waveform_gen_sep if waveform_gen_sep is not None else waveform_gen
        self.N_traj = N_traj
        self.verbose = verbose

        self.gwf = gwf
        self.dt = gwf.dt
        self.T = gwf.T   # T_max

        self.traj = getattr(waveform_gen.waveform_generator, 'inspiral_generator', None)
        self.amp = getattr(waveform_gen.waveform_generator, 'amplitude_generator', None)
        self.interpolate_mode_sum = getattr(waveform_gen.waveform_generator, 'create_waveform', None)
        self.ylm_gen = getattr(waveform_gen.waveform_generator, 'ylm_gen', None)

        self.delta_T = self.T * YRSID_SI / self.N_traj

        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        # Generate signal at T_max
        self.signal = waveform_gen(m1, m2, a, p0, e0, xI0, dist,
                                   qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,
                                   T=self.T, dt=self.dt)
                                   
    def _pad_to_N(self, wf):
        """Zero-pad or truncate wf to match gwf.N along axis=-1."""
        N = self.gwf.N
        L = wf.shape[-1] if hasattr(wf, 'shape') else len(wf)
        if L == N:
            return wf
        elif L > N:
            return wf[..., :N]
        else:
            pad_shape = list(wf.shape)
            pad_shape[-1] = N - L
            pad = self.gwf.xp.zeros(pad_shape, dtype=wf.dtype)
            return self.gwf.xp.concatenate([wf, pad], axis=-1)

    def _generate_selected_waveforms(self, params, selected_labels, T_prop):
        """
        Generate waveforms for selected mode groups at T_prop, padded to gwf.N.

        Returns array of shape (N_samples, N_groups).
        """
        m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0 = params

        waveforms_per_group = []
        for group in selected_labels:
            waveform_group = self.waveform_gen(
                m1, m2, a, p0, e0, xI0, dist,
                qS, phiS, qK, phiK,
                Phi_phi0, Phi_theta0, Phi_r0,
                dt=self.dt,
                T=T_prop,
                mode_selection=group,
                include_minus_mkn=False,
            )
            waveforms_per_group.append(self._pad_to_N(waveform_group))

        return self.gwf.xp.stack(waveforms_per_group, axis=1)

    def __call__(self, theta_template):
        """
        theta_template: [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
                         Phi_phi0, Phi_theta0, Phi_r0, T]
        T is the proposed observation time in years (last element).
        """
        *phys_params, T_prop = theta_template


        m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t, qS_t, phiS_t, qK_t, phiK_t, Phi_phi0_t, Phi_theta0_t, Phi_r0_t = phys_params
        

        h_temp_raw = self.waveform_gen(
            m1_t, m2_t, a_t, p0_t, e0_t, xI0_t, dist_t,
            qS_t, phiS_t, qK_t, phiK_t,
            Phi_phi0_t, Phi_theta0_t, Phi_r0_t,
            dt=self.dt,
            T=T_prop,
        )
        N_len = len(h_temp_raw)
        h_temp = self._pad_to_N(h_temp_raw)
        h_signal = self._pad_to_N(self.signal[:N_len])

        snr_signal = self.gwf.rhostat_timemax(h_signal)

        rho_tot = self.gwf.rhostat_timemax(h_temp)
        X_scalar = self.gwf.Xstat_timemax(h_signal, h_temp)

        if self.verbose:
            print(f'rho_tot: {rho_tot:.4g}')
            print(f"X_scalar (time-maximized): {X_scalar:.4g}")

        overlap = X_scalar / snr_signal

        if hasattr(overlap, 'get'):
            logl_res = float(overlap.get())
        else:
            logl_res = float(overlap)

        if self.verbose:
            print(f"Overlap: {logl_res:.6g}")

        return logl_res
