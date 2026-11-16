"""
piv_speaker.py
==============
PIV post-processing for the embedded-speaker (USound UT-P2020) measurements on
the NACA 65(1)-412 wing.  Extracted from the "Class and function-definitions"
cell of Classes.ipynb.

Classes
-------
PIVSpeakerPhaseAverage
    One LaVision phase-averaged (or standard-deviation) PIV set per run:
    loading, wall-circle fit, surface-conformed coordinates, Gaussian fits of
    u_n across the jet, and momentum/flux integrals.
PIVSpeakerInterlaced
    The 8 interlaced phases of one run (one PIVSpeakerPhaseAverage per phase),
    cropped to a common grid.  u_n_mean is the steady (streaming) field and
    u_n_fluc_rms the RMS of the coherent oscillation (std over the 8 phases).

Units: velocities in m/s; coordinates x, y, X, Y, rho normalised by chord c.

Requires lvpyio (LaVision reader) only for load_run; it is imported lazily so
the module can be imported without it.
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.optimize

try:  # package or flat import (Colab adds src/ to sys.path)
    from . import streaming_scaling as _ss
except ImportError:
    import streaming_scaling as _ss

try:
    import lvpyio as lv
except ImportError:  # pragma: no cover - only needed for load_run
    lv = None

RHO = 1.18  # default air density [kg/m^3]; override per instance with rho0=


class PIVSpeakerPhaseAverage:
    MI_ROOT = r"/content/drive/MyDrive/Wind Tunnel Shared/Experimental Campaigns/24-01_Naca_651412_Campaign/Model Installation/InstallationRunFolders"
    PIV_ROOT = r"/content/drive/MyDrive/Wind Tunnel Shared/Experimental Campaigns/24-01_Naca_651412_Campaign/PIV/N65_WT_PIV_Flow_Fields"

    def __init__(self, rho0=RHO):
        """
        Parameters
        ----------
        rho0 : float
            Air density [kg/m^3] used in momentum_integral.  Default 1.18.
            For lab conditions use e.g.
            acoustic_properties.air_properties(T_C, RH_pct, P_Pa)["rho"].
            (Named rho0 because self.runs[run]["rho"] is the wall distance.)
        """
        self.rho0 = rho0
        self.runs = {}
        self.u_lim = 8  # colour limit [mm/s]; plot_u overwrites it

    def load_run(self, run, avg=True, cb_size=8, undscore=None):
        # initialize run dict
        self.avg = avg

        self.runs[run] = {}

        # name of lavision project containing PIV data
        if run < 2150:
            project_run = 1387
        elif run< 2225:
            project_run = 2173
        else:
            project_run = 2225
        project = f"CJK_NACA651412_RUN{project_run}"

        project_path = os.path.join(self.PIV_ROOT, project)

        # photo directory
        photo_dir = os.path.join(self.MI_ROOT, f"Run{project_run}")

        c = 0.2
        # dh = 0.5e-3  # m

        # Run number for processed PIV
        #identify folder corresponding to run (regardless of TP number)
        run_dir = next(f for f in os.listdir(project_path) if f.startswith(f"RUN{run}") and not f.endswith(".set"))

        run_path = os.path.join(project_path, run_dir)
        if undscore is not None:
            suffix = f"_{undscore:02d}"
        else:
            suffix = ""


        if lv is None:
            raise ImportError("lvpyio is required to load PIV data: pip install lvpyio==1.3.1")

        found = False
        for s in ('_', '%'):
            inst_piv_path = os.path.join(run_path, f"PIV_MP(3x{cb_size}x{cb_size}_50{s}ov_ImgCorr)"+suffix)
            avgd_piv_path = os.path.join(inst_piv_path, "Avg_Stdev")
            if os.path.exists(avgd_piv_path):
                piv_avg_b, piv_std_b = lv.read_set(avgd_piv_path)  # set has 2 buffers
                found = True
                break
        assert found

        if self.avg:
            piv_b = piv_avg_b
        else:
            piv_b = piv_std_b

        scale = piv_b[0].scales.x.slope
        scale_unit = piv_b[0].scales.x.unit

        piv_avg = piv_b.as_masked_array()

        mask = piv_avg.mask["u"]
        indices = smallestbox(~mask)
        mask = mask[indices]

        u = piv_avg.data["u"]
        v = piv_avg.data["v"]

        u = u[indices]
        v = v[indices]


        frame = piv_b[0]

        height, width = u.shape
        scales = [frame.scales.x, frame.scales.y]

        coordinates_x = frame.grid.x * np.arange(width) + frame.grid.x // 2
        coordinates_y = frame.grid.y * np.arange(height) + frame.grid.y // 2
        coordinates_x =  scales[0].slope * coordinates_x + scales[0].offset
        coordinates_y =  scales[1].slope * coordinates_y + scales[1].offset

        units = [scale.unit for scale in scales]

        for i, coord in enumerate([coordinates_x, coordinates_y]):
            if units[i] == 'mm':  # convert to m
                coord *= 1e-3
            coord /= c  # normalize

        x = coordinates_x
        y = coordinates_y


        # flip  coordinate
        x = -x
        u = -u

        u_mag = np.sqrt(u**2 + v**2)

        if run < 1470:
            x += 0.1
            y += 0.14
        elif run < 1600:
            x += 0.02
            y += 0.1465
        elif run < 2224:
            y += 0.025
            x += -0.021
        elif run < 2246:
            y += -0.05
            x += -0.044

        X, Y = np.meshgrid(x, y)

        self.runs[run]["mask"] = mask
        self.runs[run]["u"] = u
        self.runs[run]["v"] = v
        self.runs[run]["u_mag"] = u_mag

        self.runs[run]["x"] = x
        self.runs[run]["y"] = y

        self.runs[run]["width"] = width
        self.runs[run]["height"] = height

        self.runs[run]["X"] = X
        self.runs[run]["Y"] = Y

    def vorticity(self, run):
        u = self.runs[run]["u"]
        v = self.runs[run]["v"]

        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        print(np.shape(x), np.shape(y))
        print(np.shape(u))

        dvdx = np.gradient(v, x, axis=1)
        dudy = np.gradient(u, y, axis=0)

        omega = dvdx - dudy

        self.runs[run]["omega"] = omega

    def fit_circle(self, run, fit_region_prop=None):
        mask = self.runs[run]["mask"]
        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        width = self.runs[run]["width"]
        height = self.runs[run]["height"]

        if fit_region_prop is None:
            if 2228 <= run <= 2245:
                fit_region_prop = 0.98
            else:
                fit_region_prop = 0.9

        edge_prop = (1-fit_region_prop)/2
        ix_min = int(edge_prop*width)
        iy_min = int(edge_prop*height)
        x_fit = x[ix_min:-ix_min]
        y_fit = y[iy_min:-iy_min]
        mask_fit = mask[iy_min:-iy_min,ix_min:-ix_min]

        # plt.pcolormesh(mask_fit)
        # plt.show()
        X_fit, Y_fit = np.meshgrid(x_fit, y_fit)

        circle_to_fit = lambda x, xc, yc, r: np.sqrt(r**2 - (x-xc)**2) + yc

        xp = X_fit[mask_fit].flatten()
        yp = Y_fit[mask_fit].flatten()

        xg = np.mean(x)
        yg = np.min(y)
        rg = np.max(x)-np.min(x)

        xc, yc, r = scipy.optimize.curve_fit(circle_to_fit, xp, yp, (xg, yg, rg))[0]

        # account for thickness of mask
        r_mask = np.sqrt((xp-xc)**2 + (yp-yc)**2)
        r_std = np.std(r_mask)
        if run < 1470:
            factor = 3.3
            npx = 0
        elif run < 1520:
            factor = 4.5
            npx = 0
        elif run < 2246:
            factor = 4
            npx = 2
        else:
            factor = 4
            npx = 0

        dy = abs(y[1] - y[0])
        buffer = max(factor*r_std, npx*dy)
        r += buffer

        circle = lambda x: circle_to_fit(x, xc, yc, r)

        self.runs[run]["xc"] = xc
        self.runs[run]["yc"] = yc
        self.runs[run]["r"] = r
        self.runs[run]["circle"] = circle

    def theta_to_xi(self, run, phi):
        return self.runs[run]['r']*self.runs[run]['theta0'] - phi

    def xi_to_theta(self, run, xi):
        return self.runs[run]['theta0'] - xi/self.runs[run]['r']

    def surface_conformed_coordinates(self, run):
        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]
        xc = self.runs[run]["xc"]
        yc = self.runs[run]["yc"]
        r = self.runs[run]["r"]
        x = self.runs[run]["x"]
        y = self.runs[run]["y"]
        u = self.runs[run]["u"]
        v = self.runs[run]["v"]
        u_mag = self.runs[run]["u_mag"]

        X_shifted = X - xc
        Y_shifted = Y - yc

        mask_circ = np.sqrt((X_shifted)**2 + (Y_shifted)**2) >= r

        rho = np.sqrt(X_shifted**2 + Y_shifted**2) - r
        theta = np.arctan2(Y_shifted, X_shifted)

        self.runs[run]['theta0'] = np.max(theta)

        theta2xi = lambda phi:  r*(np.max(theta) - phi)
        xi2theta = lambda xi: np.max(theta) - xi/r

        xi = self.theta_to_xi(run, theta)

        u_n = (u*X_shifted+v*Y_shifted)/(rho+r)

        u *= mask_circ
        v *= mask_circ
        u_n *= mask_circ
        u_mag *= mask_circ


        self.runs[run]["rho"] = rho
        self.runs[run]["theta"] = theta
        self.runs[run]["xi"] = xi
        self.runs[run]["u"] = u
        self.runs[run]["v"] = v
        self.runs[run]["u_n"] = u_n
        self.runs[run]["u_mag"] = u_mag

    def peak_rms(self, run, etamin=5e-3, etamax=15e-3, mode="u_n'"):
        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]
        rho = self.runs[run]["rho"]  # radial distance from surface

        if mode == "u_n'":
            u_arr = self.runs[run]["u_n"]
            comp = "u_n"
            cvar = "u_n'-u_{n,b}'"
        elif mode == "u_mag'":
            u_arr = self.runs[run]["u_mag"]
            comp = "u_mag"
            cvar = r"\|u'\|'"


        masked_u = u_arr*np.all([etamin <= rho, rho <= etamax, X <= 0.02, X >= -0.02], axis=0)
        u_arr -= (np.median(masked_u[masked_u > 0]))
        peak_rms_coords = np.argmax(masked_u.flatten())
        peak_rms = u_arr.flatten()[peak_rms_coords]
        peak_rms_eta = rho.flatten()[peak_rms_coords]
        peak_rms_x = X.flatten()[peak_rms_coords]
        peak_rms_y = Y.flatten()[peak_rms_coords]

        self.plot_u(run, comp=comp, show=False, cvar=cvar)
        plt.scatter(peak_rms_x, peak_rms_y, c="k", s=10)
        plt.show()

        return peak_rms, peak_rms_eta#, (peak_rms_x, peak_rms_y)

    def magnitude_identification(self, run, etamin=5e-3, etamax=15e-3, netas=5):
        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        xc = self.runs[run]["xc"]
        yc = self.runs[run]["yc"]

        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]
        ur = self.runs[run]["u_n"]
        xi = self.runs[run]["xi"]
        theta = self.runs[run]["theta"]
        r = self.runs[run]["r"]

        # define ur for interpolation
        ur_cmap = matplotlib.colormaps['spring']

        cmap = plt.colormaps['seismic']
        ticks = np.linspace(-self.u_lim, self.u_lim, 8, endpoint=True)
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=ticks,
            ncolors=256,
            extend='both',
        )

        ur_interp = scipy.interpolate.RegularGridInterpolator((x, y), ur.T, fill_value=np.nan, bounds_error=False)

        theta_arr = np.linspace(np.min(theta), np.max(theta), 100)
        theta_deg_arr = np.rad2deg(theta_arr)
        etavalues = np.linspace(etamin, etamax, netas)

        def color(eta):
            # put eta between 0 and 1
            eta_scaled = (eta-np.min(etavalues))/(np.max(etavalues)-np.min(etavalues))
            # remove last 10% which is too yellow
            eta_scaled = eta_scaled*0.9
            return ur_cmap(eta_scaled)

        xwid_fig_ur_xi = 0.03

        # fit Gaussian of amplitude a, mean mu, and std sigma summed with 2nd order polynomial
        gaus_third_poly_w_params = lambda x, a, mu, sigma, b, m: a*np.exp(-(x-mu)**2/(2*sigma**2)) + b + m*x #+ q*x**2 + c*x**3

        # plt.figure(2)
        # plt.title(r"$u_n(x)$")
        # plt.xlabel(r"$x/c$")
        # plt.ylabel("$u_n$")
        # # plt.xlim(0, xwid_fig_ur_xi)
        # plt.xlim([-0.02, 0.02])

        # plt.figure(1)
        # plt.pcolormesh(X, Y, ur, cmap=cmap, norm=norm)
        # plt.colorbar(ticks=ticks, label=f"$u_n$ [m/s]")
        # if "xc" in self.runs[run]:
        #     circle = self.runs[run]["circle"]
        #     plt.fill(np.concatenate((x, x[::-1])), np.concatenate((circle(x), np.min(y)*np.ones(len(x)))), 'grey')
        # plt.title(r"Contours along which to extract $u_n(x)$")
        y_0fig_ur_xi = 0.005
        # plt.xlim(0, xwid_fig_ur_xi)
        # plt.ylim(y_0fig_ur_xi, y_0fig_ur_xi+xwid_fig_ur_xi)
        # plt.xlabel("$x/c$")
        # plt.ylabel("$y/c$")
        # plt.ylim([0, 0.04])
        xmin = -0.02
        xmax = 0.02
        # plt.xlim([xmin, xmax])
        xplotarr = np.linspace(xmin, xmax, 500)


        avalues = np.zeros_like(etavalues)
        aerrvalues = np.zeros_like(etavalues)
        mu_fit_std_values = np.zeros_like(etavalues)
        mu_fit_values = np.zeros_like(etavalues)

        for i, eta in enumerate(etavalues):
            r_eta = eta + r

            x_arr, y_arr = r_eta*np.cos(theta_arr)+xc, r_eta*np.sin(theta_arr)+yc
            xi_arr = self.theta_to_xi(run, theta_arr)
            ur_arr = ur_interp((x_arr, y_arr))

            good_mask = np.logical_and(np.logical_not(np.isnan(ur_arr)), ur_arr != 0)

            x_arr = x_arr[good_mask]
            y_arr = y_arr[good_mask]
            xi_arr = xi_arr[good_mask]
            ur_arr = ur_arr[good_mask]

            if np.sum(good_mask) <= 7:
                print("fewer than 7 usable points")
                continue

            if run < 1470:
                x0 = 0
            elif run < 1520:
                x0 = 0
            else:
                x0 = 0

            urpopt, urpcov = scipy.optimize.curve_fit(
                gaus_third_poly_w_params,
                x_arr,
                ur_arr,
                (0, x0, 0.01, 0, 0),#, 0, 0),
                maxfev=1e4,
                bounds=(
                    (-1e-1, np.min(x_arr), 0, -np.inf, -np.inf),#, -np.inf, -np.inf),
                    (1e-1, np.max(x_arr), (np.max(x_arr)-np.min(x_arr))/2, np.inf, np.inf),#, np.inf, np.inf),
                ),
            )
            perr = np.sqrt(np.diag(urpcov))
            a_fit, mu_fit, sigma_fit, b_fit, m_fit= urpopt
            a_fit_std, mu_fit_std, sigma_fit_std, b_fit_std, m_fit_std = perr

            gaus_third_poly_w_params_fit = lambda x: gaus_third_poly_w_params(x, a_fit, mu_fit, sigma_fit, b_fit, m_fit)#, q_fit, c_fit)

            avalues[i] = a_fit
            aerrvalues[i] = a_fit_std
            mu_fit_std_values[i] = mu_fit_std
            mu_fit_values[i] = mu_fit

            # if np.abs(a_fit/a_fit_std) > 5 and np.abs(mu_fit_std)<1e-3:  # mus_std was 5e-4
                # print(f'eta: {eta:1.2e}, a: {a_fit:1.2e}, xbar: {mu_fit:1.2e} +/- {mu_fit_std:1.1e}, sigma: {sigma_fit:1.2e}, p/n: {a_fit/a_fit_std:1.1e}')

                # plt.figure(1)  # contours overlaid on ur
                # plt.plot(x_arr, y_arr, "-", c=color(eta))

                # plt.figure(2)  # ur(xi)
                # plt.plot(x_arr, ur_arr, ".", c=color(eta))
                # plt.plot(xplotarr, gaus_third_poly_w_params_fit(xplotarr), '--', c=color(eta))

        avalues[np.logical_or(avalues==0, aerrvalues==0)] = np.nan
        aerrvalues[np.logical_or(avalues==0, aerrvalues==0)] = np.nan

        median_mu = np.median(mu_fit_values)

        realas = np.logical_and(
            np.abs(mu_fit_std_values) < 1e-3,
            # np.abs(mu_fit_values - median_mu) < 1e2*mu_fit_std_values,
            np.abs(avalues/aerrvalues) > 5
        )

        if np.sum(realas)>=1:
            maxidx = np.argmax(np.abs(avalues[realas]))
            amax = avalues[realas][maxidx]
            eta_amax = etavalues[realas][maxidx]
        else:
            # No fit passed the filters.  amax = 0 used to be returned with
            # only print(realas), which made a weak (e.g. low-f) run look like
            # "zero amplitude" downstream.  Warn loudly instead.
            with np.errstate(divide="ignore", invalid="ignore"):
                sn = np.abs(avalues / aerrvalues)
            best_sn = np.nanmax(sn) if np.any(np.isfinite(sn)) else np.nan
            warnings.warn(
                f"magnitude_identification(run={run}): no eta level passed the "
                f"filters (|a/a_err| > 5 and mu_std < 1e-3; best |a/a_err| = "
                f"{best_sn:.2g} over {netas} levels in [{etamin:g}, {etamax:g}]). "
                f"Returning amax = 0, eta_amax = nan: this is 'not detected', "
                f"not a measured zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            amax = 0
            eta_amax = np.nan
        # print(f"eta: {etamax:1.2e}, a: {amax:1.2e}")

        # plt.show()


        # plt.scatter(etavalues[realas], avalues[realas], c=color(etavalues[realas]), cmap=ur_cmap)
        # plt.xlabel(r"$\eta$")
        # plt.ylabel(r"$u_e$ [m/s]")
        # plt.show()

        return amax, eta_amax

    def fit_dual_gaussians(self, run, etamin=5e-3, etamax=15e-3, netas=5, p0=None):
        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        xc = self.runs[run]["xc"]
        yc = self.runs[run]["yc"]

        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]
        ur = self.runs[run]["u_n"]
        xi = self.runs[run]["xi"]
        theta = self.runs[run]["theta"]
        r = self.runs[run]["r"]

        # define ur for interpolation
        ur_cmap = matplotlib.colormaps['spring']

        theta_arr = np.linspace(np.min(theta), np.max(theta), 200)

        mumin = np.min(theta_arr)
        mumax = np.max(theta_arr)
        sigmamax = (mumax-mumin)/15

        umax = 3e-1

        cmap = plt.colormaps['seismic']
        ticks = np.linspace(-self.u_lim, self.u_lim, 8, endpoint=True)
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=ticks,
            ncolors=256,
            extend='both',
        )

        ur_interp = scipy.interpolate.RegularGridInterpolator((x, y), ur.T, fill_value=np.nan, bounds_error=False)

        etavalues = np.linspace(etamin, etamax, netas)

        if run < 1470:
            theta0 = 1.638
        elif run < 1520:
            theta0 = 1.638
        else:
            theta0 = np.pi/2

        def color(eta):
            # put eta between 0 and 1
            eta_scaled = (eta-np.min(etavalues))/(np.max(etavalues)-np.min(etavalues))
            # remove last 10% which is too yellow
            eta_scaled = eta_scaled*0.9
            return ur_cmap(eta_scaled)

        xwid_fig_ur_xi = 0.03

        plt.figure(2)
        plt.title(r"$u_n(\theta)$")
        plt.xlabel(r"$\theta$")
        plt.ylabel("$u_n$")

        plt.figure(1)
        plt.pcolormesh(X, Y, ur, cmap=cmap, norm=norm)
        plt.colorbar(ticks=ticks)
        if "xc" in self.runs[run]:
            circle = self.runs[run]["circle"]
            plt.fill(np.concatenate((x, x[::-1])), np.concatenate((circle(x), np.min(y)*np.ones(len(x)))), 'grey')
        plt.title(r"Contours along which to extract $u_n(\xi)$")
        y_0fig_ur_xi = 0.005
        plt.xlabel("$x/c$")
        plt.ylabel("$y/c$")

        urpopts = []
        perrs = []
        fit_funcs = []
        rms_errors = []

        def null_return():
                urpopt, urpcov = np.zeros(9), np.zeros((9,9))
                # replace 0s with nans
                urpopt[:], urpcov[:] = np.nan, np.nan
                return urpopt, urpcov

        # parameters in order:
        #                   mu,        a, fa,      sigma, fsigma, b, m, q, c
        p_guess_init = (theta0, -umax/10,  0, sigmamax/3,    1.5, 0, 0, 0, 0),
        p_guess = p_guess_init[0] if p0 is None else p0

        for i, eta in enumerate(etavalues):
            theta_arr = np.linspace(np.min(theta), np.max(theta), 200)
            r_eta = eta + r

            x_arr, y_arr = r_eta*np.cos(theta_arr)+xc, r_eta*np.sin(theta_arr)+yc
            ur_arr = ur_interp((x_arr, y_arr))

            data_mask = np.logical_and(np.logical_not(np.isnan(ur_arr)), ur_arr != 0)

            ur_arr = ur_arr[data_mask]
            theta_arr = theta_arr[data_mask]

            # truncate first and last datapoint of ur_arr and theta_ar to avoid null interpolation contamination
            ur_arr = ur_arr[1:-1]
            theta_arr = theta_arr[1:-1]

            if np.sum(data_mask) <= 7:
                urpopt, urpcov = null_return()
                print("fewer than 7 usable points")
                continue

            try:
                # parameters in order:
                #        mu,        a,  fa,    sigma, fsigma,       b,       m,       q,       c
                urpopt, urpcov = scipy.optimize.curve_fit(
                    dual_gaus_third_poly_w_params,
                    theta_arr,
                    ur_arr,
                    p_guess,
                    maxfev=1e4,
                    bounds=(
                        (mumin, -umax,   0,        0,    1.5, -np.inf, -np.inf, -np.inf, -np.inf),
                        (mumax,  umax, 0.7, sigmamax,      5,  np.inf,  np.inf,  np.inf,  np.inf),
                    ),
                )
            except RuntimeError: # except error if fit fails to converge
                urpopt, urpcov = null_return()
                print("fit failed to converge")

            perr = np.sqrt(np.diag(urpcov))

            urpopts.append(urpopt)
            perrs.append(perr)

            mu, a, fa, sigma, fsigma, b, m, q, c = urpopt
            mu_err, a_err, fa_err, sigma_err, fsigma_err, b_err, m_err, q_err, c_err = perr

            a1, a2 = a, a*fa
            sigma1, sigma2 = sigma, sigma*fsigma

            print(f'eta: {eta:3g}, xbar: {mu:3g} +/- {mu_err:3g}, a1, a2: {a1:3g}, {a2:3g}, s/n: {a/a_err:2g}, sigma1, 2: {sigma1:3g}, {sigma2:2g}')


            if np.abs(a/a_err) < 2:
                a = 0
                urpopt[1] = 0
                a1, a2 = 0, 0

            if np.abs(fa/fa_err) < 2:
                urpopt[2] = 0
                a2 = 0

            dual_gaus_third_poly_fit = lambda theta: dual_gaus_third_poly_w_params(theta,  mu, a, fa, sigma, fsigma, b, m, q, c)
            fit_funcs.append(dual_gaus_third_poly_fit)

            rms_errors.append(np.sqrt(np.mean((ur_arr - dual_gaus_third_poly_fit(theta_arr))**2)))

            plt.figure(1)  # contours overlaid on ur
            plt.plot(x_arr, y_arr, "-", c=color(eta))

            plt.figure(2)  # ur(xi)
            plt.plot(theta_arr, ur_arr, "-", c=color(eta))
            plt.plot(theta_arr, dual_gaus_third_poly_fit(theta_arr), '--', c=color(eta))

        plt.show()

        # numpy array the param arrays
        urpopts = np.array(urpopts)
        perrs = np.array(perrs)

        # take average and standard deviation of each param
        urpopt_avg = np.nanmean(urpopts, axis=0)
        urpopt_std = np.nanstd(urpopts, axis=0)

        # form param array with means subtracted, normalized by standard deviations
        urpopts_deviations = np.abs((urpopts - urpopt_avg)/urpopt_std)
        deviation_mask = urpopts_deviations[:,0] < 1.5

        rms_errors = np.array(rms_errors)
        rms_err_mask = rms_errors < 1.2e-4

        a_amp_mask = np.abs(urpopts[:, 1]) > 0.1*np.max(np.abs(urpopts[:, 1]))
        sigma_mask = np.abs(urpopts[:, 3]) < 5*np.min(np.abs(urpopts[:, 3]))

        fits_mask = np.all([deviation_mask, rms_err_mask, a_amp_mask], axis=0)

        # redefine ur_popt_avg to be average across etas with fits_mask true
        urpopt_avg = np.mean(urpopts[fits_mask], axis=0)
        urpopt_std = np.std(urpopts[fits_mask], axis=0)

        print(rms_errors)
        print(urpopts_deviations[:,0])
        print(deviation_mask)
        print(rms_err_mask)
        print(fits_mask)

        plt.figure(3)
        for i, (urpopt, perr) in enumerate(zip(urpopts, perrs)):
            # if not fits_mask[i]:
            #     continue
            r_eta = etavalues[i] + r
            x_arr, y_arr = r_eta*np.cos(theta_arr)+xc, r_eta*np.sin(theta_arr)+yc
            ur_arr = ur_interp((x_arr, y_arr))

            mu, a1, a2, sigma1, sigma2, b, m, q, c = urpopt
            mu_err, a1_err, a2_err, sigma1_err, sigma2_err, b_err, m_err, q_err, c_err = perr

            dual_gaus_third_poly_fit = lambda theta: dual_gaus_third_poly_w_params(theta, mu, a1, a2, sigma1, sigma2, b, m, q, c)

            plt.plot(theta_arr, dual_gaus_third_poly_fit(theta_arr), '--', c=color(etavalues[i]))
            plt.plot(theta_arr, ur_interp((x_arr, y_arr)), "-", c=color(etavalues[i]))

        plt.xlabel("$\\theta$")
        plt.ylabel("$u_n$")
        plt.title(r"$u_n(\theta)$, only good fits")
        plt.show()

        return urpopt_avg, fits_mask, urpopts

    def momentum_integral(self, run, etamin=5e-3, etamax=15e-3, netas=5):
        """Flux of the fitted u_n jet through spherical caps, integrated over eta.

        ASSUMES AXISYMMETRY.  The PIV plane only gives u_n(theta) in one
        plane through the speaker axis.  Each dual-Gaussian fit u_n(alpha)
        (alpha = angle from the axis) is taken to be rotationally symmetric
        about that axis and integrated over a spherical cap of radius
        r = r_wall + eta:

            m(eta) = rho0 * 2 pi r^2 * int_0^{alpha_max} u_n(alpha) sin(alpha) d(alpha)

        with alpha_max = pi/24.  Then int m(eta) d(eta) is taken with Simpson's
        rule.  Any asymmetry of the jet (e.g. from the
        wall curvature or background flow) is not captured.

        Notes
        -----
        * The integrand is linear in u_n, so m(eta) is a mass flux, not a
          momentum flux (rho u^2).
        * Lengths are chord-normalised (x/c), so the result is in units of
          kg/s * c^3; multiply by c^3 for SI.
        """
        etavalues = np.linspace(etamin, etamax, netas)

        urpopt_avg, fits_mask, urpopts = self.fit_dual_gaussians(run, etamin=etamin, etamax=etamax, netas=netas)
        print(fits_mask)

        # run fit_dual_gaussians again with initial guess given by average parameters of previous call
        # urpopt_avg, fits_mask, urpopts = self.fit_dual_gaussians(run, etamin=etamin, etamax=etamax, netas=netas, p0=urpopt_avg)
        masked_eta = etavalues[fits_mask]

        # fits_mask = np.ones_like(fits_mask)

        # plot amplitudes a and sigmas against eta on two y axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        a_arr = urpopts[fits_mask, 1]
        fa_arr = urpopts[fits_mask, 2]
        sigma_arr = urpopts[fits_mask, 3]
        fsigma_arr = urpopts[fits_mask, 4]
        a1_arr = a_arr
        a2_arr = a1_arr*fa_arr
        sigma1_arr = sigma_arr
        sigma2_arr = sigma_arr*fsigma_arr

        ax1.plot(masked_eta, a1_arr, label="$a_1$", color="red")
        ax1.plot(masked_eta, a2_arr, label="-$a_2$", color="orange")
        ax2.plot(masked_eta, sigma1_arr, label="$\\sigma_1$", color="blue")
        ax2.plot(masked_eta, sigma2_arr, label="$\\sigma_2$", color="purple")
        ax1.set_xlabel("$\\eta$")
        ax1.set_ylabel("$a$")
        ax2.set_ylabel("$\\sigma$")
        # draw legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)

        plt.show()

        # plot fa and fsigma on separate y axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(masked_eta, fa_arr, label="$f_a$", color="red")
        ax2.plot(masked_eta, fsigma_arr, label="$f_\\sigma$", color="blue")
        ax1.set_xlabel("$\\eta$")
        ax1.set_ylabel("$f_a$")
        ax2.set_ylabel("$f_\\sigma$")
        # draw legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2)
        plt.show()

        alpha_max = np.pi/24
        alpha_arr = np.linspace(0, alpha_max, 100)

        a0 = np.abs(a_arr[0])
        shell_momenta = np.zeros(len(masked_eta))
        # determine momentum per length of each spherical shell with circular Gaussian fit
        for i, urpopt in enumerate(urpopts[fits_mask]):
            r = masked_eta[i] + self.runs[run]["r"]

            _, a, fa, sigma, fsigma, _, _, _, _ = urpopt
            mu, b, m, q, c, = 0, 0, 0, 0, 0

            if np.abs(a) < 0.2*a0:
                break

            # assume all Gaussians are centered at point theta0 such that alpha = theta - theta0
            jet = lambda alpha: dual_gaus_third_poly_w_params(alpha, mu, a, fa, sigma, fsigma, b, m , q, c)
            integrand_theta = lambda alpha: jet(alpha)*np.sin(alpha)
            plt.plot(alpha_arr, jet(alpha_arr))

            # integral in beta gives factor of 2 pi (integrating around the spherical cap)
            # integrating in alpha needs to be evaluated numerically (azimuth), from 0 to pi/2
            shell_momenta[i] = 2 * np.pi * r**2 * scipy.integrate.quad(integrand_theta, 0, alpha_max)[0] * self.rho0
            # if i and np.sign(shell_momenta[i]) != np.sign(shell_momenta[i-1]):
            #     shell_momenta[i] = 0
            #     break
        plt.show()

        plt.plot(masked_eta, shell_momenta)
        plt.plot(masked_eta, np.zeros_like(masked_eta), '--g')
        plt.xlabel(r"$\eta$")
        plt.ylabel(r"$m(\eta)$")
        plt.show()

        momentum = scipy.integrate.simpson(shell_momenta, masked_eta)
        print(momentum)
        return momentum

    def momentum_coefficient(self, runs, etamin=5e-3, etamax=15e-3, netas=5):
        phases = np.array([0, 30, 60, 90, 180, 270])[:len(runs)]
        momenta = np.zeros(len(runs))
        for i, run in enumerate(runs):
            momentum = self.momentum_integral(run, etamin=etamin, etamax=etamax, netas=netas)
            momenta[i] = momentum
        print(momenta)
        # fit sinusoid phase and amplitude to curve
        popt, pcov = scipy.optimize.curve_fit(
            lambda x, a, phi: a*np.sin(np.deg2rad(x)+phi),
            phases,
            momenta,
        )

        plt.scatter(phases, momenta)
        # plot sinusoid over angle range
        plt.plot(np.linspace(0, 360, 100), popt[0]*np.sin(np.deg2rad(np.linspace(0, 360, 100))+popt[1]))

        plt.xlabel("$\\phi$ [deg]")
        plt.ylabel("momentum")
        plt.show()

    def planar_gaussians(self, run):
        theta = self.runs[run]["theta"]
        xc = self.runs[run]["xc"]
        yc = self.runs[run]["yc"]
        r = self.runs[run]["r"]
        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]

        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        piv_mask = self.runs[run]["mask"]
        # invert piv_mask
        piv_mask = ~piv_mask

        ur_interp = scipy.interpolate.interp2d(x, y, self.runs[run]["u_n"])

        R = np.sqrt(X**2 + Y**2)
        radius_mask = R > r

        mask = np.all(radius_mask, piv_mask)

        X_fit = X[mask]
        Y_fit = Y[mask]

        ur_mask = ur_interp(X_fit, Y_fit)

        #flatten X_fit and Y_fit to 2xN array for curve_fit
        indep = np.array([X_fit.flatten(), Y_fit.flatten()])

        # flatten ur_mask for curve_fit
        dep = ur_mask.flatten()

    def plot_u(self, run, u_lim=8, comp='v', show=True, xlims=(-0.02, 0.02), ylims=(0, 0.04), cvar=None, title=None, quiver=False, u=None, v=None, dep_var=None, show_axis=True):
        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]
        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        xc = self.runs[run]["xc"]
        yc = self.runs[run]["yc"]
        r = self.runs[run]["r"]

        self.u_lim = u_lim

        if comp == "u_mag":
            cvar = r"\|u\|"
        if comp == "u'_mag":
            cvar = r"\mathrm{sign}(\overline{v})\|u\|"

        if cvar is None:
            cvar = comp

        ticks = np.linspace(-u_lim, u_lim, 40, endpoint=True)
        labelticks = np.linspace(-u_lim, u_lim, 9, endpoint=True)
        cmap = plt.colormaps['seismic']
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=ticks,
            ncolors=256,
            extend='both',
        )

        if xlims:
            plt.xlim(xlims)
        if ylims:
            plt.ylim(ylims)

        if not(u is not None and v is not None and dep_var is not None):
            u = self.runs[run]["u"]
            v = self.runs[run]["v"]
            dep_var = self.runs[run][comp]


        plt.pcolormesh(X, Y, dep_var*1e3, cmap=cmap, norm=norm)
        plt.colorbar(ticks=labelticks, label=f"${cvar}$ [mm/s]")
        # if not title:
        #     plt.title(f'${comp}$-velocity field')
        # else:
        #     plt.title(title)
        plt.xlabel("$x/c$")
        plt.ylabel("$y/c$")

        if quiver:
            plt.quiver(x, y, u, v)

        if "xc" in self.runs[run]:
            circle = self.runs[run]["circle"]
            plt.fill(np.concatenate((x, x[::-1])), np.concatenate((circle(x), np.min(y)*np.ones(len(x)))), 'grey')

        if show_axis:
            # assume axis of speaker is normal to surface (radial direction)
            # further assume it passes through x=0
            rx = -xc
            ry = np.sqrt(r**2 - rx**2)
            xs = np.array([0, rx])
            ys = np.array([0, ry]) + yc + ry

            plt.plot(xs, ys)



        if show:
            plt.show()
        else:
            return plt.gcf(), plt.gca()

    def plot_omega(self, run, omega_lim=4.2e-6):
        if "omega" not in self.runs[run]:
            self.vorticity(run)

        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]

        x = self.runs[run]["x"]
        y = self.runs[run]["y"]

        omega = self.runs[run]["omega"]

        ticks = np.linspace(-omega_lim, omega_lim, 8, endpoint=True)
        cmap = plt.colormaps['seismic']
        norm = matplotlib.colors.BoundaryNorm(
            boundaries=ticks,
            ncolors=256,
            extend='both',
        )

        plt.pcolormesh(X, Y, omega, cmap=cmap, norm=norm)
        plt.colorbar(ticks=ticks, label=r"$\omega$")
        plt.title(r"$\omega$")
        plt.xlabel("$x/c$")
        plt.ylabel("$y/c$")

        if "xc" in self.runs[run]:
            circle = self.runs[run]["circle"]
            plt.fill(np.concatenate((x, x[::-1])), np.concatenate((circle(x), np.min(y)*np.ones(len(x)))), 'grey')

        plt.show()

    def plot_fit_circle(self, run, show=True):
        mask = self.runs[run]["mask"]
        x = self.runs[run]["x"]
        y = self.runs[run]["y"]
        circle = self.runs[run]["circle"]

        plt.plot(x, circle(x), color="orange")
        plt.contourf(x, y, mask)
        plt.title("Fit circle overlaid on mask")
        plt.xlabel("$x/c$")
        plt.ylabel("$y/c$")
        plt.colorbar()
        plt.show()

        if show:
            plt.show()
        else:
            return plt.gcf()

    def plot_surface_conformed_coordinates(self, run):
        X = self.runs[run]["X"]
        Y = self.runs[run]["Y"]
        x = self.runs[run]["x"]
        rho = self.runs[run]["rho"]
        theta = self.runs[run]["theta"]
        xi = self.runs[run]["xi"]

        circle = self.runs[run]["circle"]
        plt.fill(np.concatenate((x, x[::-1])), np.concatenate((circle(x), np.min(Y)*np.ones(len(x)))), 'grey')

        rho_cs = plt.contour(X, Y, rho, cmap="cool")
        plt.clabel(rho_cs, rho_cs.levels)

        xi_cs = plt.contour(X, Y, xi, cmap="summer")
        plt.clabel(xi_cs, xi_cs.levels)

        theta_cs = plt.contour(X, Y, theta, cmap="autumn")
        plt.clabel(theta_cs, theta_cs.levels, fmt=lambda phi: f"{np.rad2deg(phi):0.0f}")

        plt.title("Polar/surface-conformed coordinates")
        plt.show()


class PIVSpeakerInterlaced:
    def __init__(self, run, n_phases, rho0=RHO):
        """
        Parameters
        ----------
        run : int          run number (phases are LaVision sets _01 ... _08)
        n_phases : int     number of interlaced phases (8)
        rho0 : float       air density [kg/m^3], passed to each phase's
                           PIVSpeakerPhaseAverage.  Default 1.18.
                           (self.rho is the wall distance, hence rho0.)
        """
        self.run = run
        self.n_phases = n_phases
        self.rho0 = rho0

    def load_run(self):
        # use above cell technique with 8 phase averages to load
        run = self.run
        v_arr = []
        u_arr = []
        u_n_arr = []
        PMEs = []
        corner_coords = []
        # load flow fields

        PME = PIVSpeakerPhaseAverage(rho0=self.rho0)
        PME.load_run(run, avg=True, cb_size=16, undscore=1)
        PME.fit_circle(run)
        PME.surface_conformed_coordinates(run)
        for i in range(8):
            PMEs.append(PIVSpeakerPhaseAverage(rho0=self.rho0))
            PMEs[i].load_run(run, avg=True, cb_size=16, undscore=i+1)
            PMEs[i].fit_circle(run)
            PMEs[i].surface_conformed_coordinates(run)

            # identify corner coordinates for each frame
            x0, xend = PMEs[i].runs[run]["x"][0], PMEs[i].runs[run]["x"][-1]
            y0, yend = PMEs[i].runs[run]["y"][0], PMEs[i].runs[run]["y"][-1]
            corner_coords.append({
                "x0": x0,
                "xend": xend,
                "y0": y0,
                "yend": yend,
            })

        x0, xend = PME.runs[run]["x"][0], PME.runs[run]["x"][-1]
        y0, yend = PME.runs[run]["y"][0], PME.runs[run]["y"][-1]
        corner_coords.append({
            "x0": x0,
            "xend": xend,
            "y0": y0,
            "yend": yend,
        })

        small_coords = {
            "x0": np.min([corner_coords[i]["x0"] for i in range(8)]),
            "xend": np.max([corner_coords[i]["xend"] for i in range(8)]),
            "y0": np.min([corner_coords[i]["y0"] for i in range(8)]),
            "yend": np.max([corner_coords[i]["yend"] for i in range(8)]),
        }

        x = PME.runs[run]["x"]
        y = PME.runs[run]["y"]
        x_slice = slice(int(np.where(x == small_coords["x0"])[0]), int(np.where(x == small_coords["xend"])[0]))
        y_slice = slice(int(np.where(y == small_coords["y0"])[0]), int(np.where(y == small_coords["yend"])[0]))
        PME.runs[run]["x"] = PME.runs[run]["x"][x_slice]
        PME.runs[run]["y"] = PME.runs[run]["y"][y_slice]
        PME.runs[run]["X"] = PME.runs[run]["X"][y_slice, x_slice]
        PME.runs[run]["Y"] = PME.runs[run]["Y"][y_slice, x_slice]
        PME.runs[run]["rho"] = PME.runs[run]["rho"][y_slice, x_slice]

        # make cropping consistent (crop to set-intersection in x and y)
        for i in range(self.n_phases):
            # crop arrays to smallest dimension, and crop to match corner coords
            # get slices of small coord in x, y for each frame
            x = PMEs[i].runs[run]["x"]
            y = PMEs[i].runs[run]["y"]
            x_slice = slice(int(np.where(x == small_coords["x0"])[0]), int(np.where(x == small_coords["xend"])[0]))
            y_slice = slice(int(np.where(y == small_coords["y0"])[0]), int(np.where(y == small_coords["yend"])[0]))

            # redefine arrays in PMEs[i]
            PMEs[i].runs[run]["x"] = PMEs[i].runs[run]["x"][x_slice]
            PMEs[i].runs[run]["y"] = PMEs[i].runs[run]["y"][y_slice]
            PMEs[i].runs[run]["X"] = PMEs[i].runs[run]["X"][y_slice, x_slice]
            PMEs[i].runs[run]["Y"] = PMEs[i].runs[run]["Y"][y_slice, x_slice]
            PMEs[i].runs[run]["v"] = PMEs[i].runs[run]["v"][y_slice, x_slice]
            PMEs[i].runs[run]["u"] = PMEs[i].runs[run]["u"][y_slice, x_slice]
            PMEs[i].runs[run]["u_n"] = PMEs[i].runs[run]["u_n"][y_slice, x_slice]

            v_arr.append(PMEs[i].runs[run]["v"])
            u_arr.append(PMEs[i].runs[run]["u"])
            u_n_arr.append(PMEs[i].runs[run]["u_n"])

        v_arr = np.array(v_arr)
        u_arr = np.array(u_arr)

        self.X = PME.runs[run]["X"]
        self.Y = PME.runs[run]["Y"]
        self.x = PME.runs[run]["x"]
        self.y = PME.runs[run]["y"]
        self.rho = PME.runs[run]["rho"]
        self.theta = PME.runs[run]["theta"]

        self.v_mean = np.mean(v_arr, axis=0)
        self.u_mean = np.mean(u_arr, axis=0)
        self.u_n_mean = np.mean(u_n_arr, axis=0)
        self.u_fluc_rms = np.std(u_arr, axis=0)
        self.v_fluc_rms = np.std(v_arr, axis=0)
        self.u_n_fluc_rms = np.std(u_n_arr, axis=0)
        self.u_mag_fluc_rms = np.sqrt(self.u_fluc_rms**2 + self.v_fluc_rms**2)

        self.fluctuation_phases = []


        # plot and subtract mean
        for i in range(self.n_phases):
            # print(i, np.shape(PMEs[i].runs[run]["v"]))
            # print(PMEs[i].runs[run]["x"][0, 0], PMEs[i].runs[run]["x"][0, 0])
            self.fluctuation_phases.append({
                "u": PMEs[i].runs[run]["u"] - self.u_mean,
                "v": PMEs[i].runs[run]["v"] - self.v_mean,
                "u_n": PMEs[i].runs[run]["u_n"] - self.u_n_mean,
            })

        self.PME = PME

    def peak_rms(self, etamin=0, etamax=0.02, mode="u_mag", plot=True):
        """Peak RMS of the coherent oscillation near the speaker.

        Searches etamin <= rho <= etamax, |X| <= 0.015 (chord-normalised).
        mode="u_mag": |u'| RMS; mode="u_n": surface-normal RMS, which is also
        stored as self.u_n_fluc_rms_peak and self.u_n_fluc_rms_peak_rho (used by
        holman_criterion).  plot=False skips the figure.

        Returns (peak_rms [m/s], rho at the peak [chord-normalised]).
        """
        X = self.X
        Y = self.Y
        rho = self.rho

        v_rms = self.v_fluc_rms
        u_rms = self.u_fluc_rms

        if mode == "u_mag":
            u_rms = self.u_mag_fluc_rms
            cvar = r"\|\tilde{u}'\|"
        elif mode == "u_n":
            u_rms = self.u_n_fluc_rms
            cvar = r"\tilde{u_n}'"

        u_rms_masked = u_rms * np.all([etamin <= rho, rho <= etamax, X <= 0.015, X >= -0.015], axis=0)

        peak_rms_index = np.argmax(u_rms_masked.flatten())
        peak_rms_x = X.flatten()[peak_rms_index]
        peak_rms_y = Y.flatten()[peak_rms_index]
        peak_rms_rho = rho.flatten()[peak_rms_index]
        peak_rms = u_rms.flatten()[peak_rms_index]

        if mode == "u_n":
            self.u_n_fluc_rms_peak = peak_rms
            self.u_n_fluc_rms_peak_rho = peak_rms_rho

        if plot:
            self.PME.plot_u(self.run, quiver=False, u=u_rms, v=v_rms, dep_var=u_rms, show=False, cvar=cvar)
            plt.scatter(peak_rms_x, peak_rms_y, color="black")
            plt.show()

        return peak_rms, peak_rms_rho

    def holman_criterion(self, f, d_hole, nu=1.5e-5):
        """Synthetic-jet formation criterion of Holman et al. (2005).

        Returns (Re, S, Re_over_S2, forms_jet).

        Uses U0_peak = sqrt(2) * self.u_n_fluc_rms_peak (the std over the 8
        phases is the RMS of the oscillation; see the qualifying exam eq. 5.5).
        If peak_rms(mode="u_n") hasn't been called yet, it is called here with
        its default search window and plot=False.

        The Reynolds number uses the velocity averaged over the expulsion
        stroke, U_bar = (2/pi) * U0_peak (Holman 2005 convention):

            Re = U_bar d / nu,   S = sqrt(omega d^2 / nu),   Re/S^2 = U_bar / (omega d)

        forms_jet = Re/S^2 > 0.16 (axisymmetric orifice).

        Parameters
        ----------
        f : float        drive frequency [Hz]
        d_hole : float   orifice diameter [m]  (NOT yet confirmed; ask Ari)
        nu : float       kinematic viscosity [m^2/s]

        Caveat
        ------
        u_n_fluc_rms_peak is the largest PIV-measured RMS within 0-0.02c
        (0-4 mm) of the wall, not the exit velocity in the hole.  With a
        0.1 mm PIV vector spacing and a sub-mm hole it most likely
        UNDERESTIMATES U0, which biases Re/S^2 low and forms_jet toward False.
        See docs/task2_orifice_findings.md.

        Reference: Holman, Utturkar, Mittal, Smith & Cattafesta (2005),
        "Formation criterion for synthetic jets", AIAA J. 43(10), 2110-2116.
        """
        if getattr(self, "u_n_fluc_rms_peak", None) is None:
            self.peak_rms(mode="u_n", plot=False)
        U0_peak = np.sqrt(2.0) * self.u_n_fluc_rms_peak
        h = _ss.holman_criterion(f, U0_peak, d_hole, nu, velocity="amplitude")
        return (float(h["Re"][0]), float(h["S"][0]),
                float(h["Re_over_S2"][0]), bool(h["forms_jet"][0]))

    def momentum_integral(self, f, width=0.04):
        """Oscillating-flux amplitude of the speaker jet, from the 8 phases.

        ASSUMES AXISYMMETRY about the speaker axis.  The line
            integrand = u_n * np.pi * np.abs(D)
        turns the planar field into an axisymmetric one: D is the signed
        distance from the axis in the PIV plane, and integrating
        pi |D| u_n over D in [-width/2, width/2] equals
        int_0^{width/2} 2 pi s u_n(s) ds (the volume flux through a disk of
        radius width/2) only if u_n(s) is the same on both sides of the axis.
        Otherwise it gives the average of the two half-planes.  Out-of-plane
        asymmetry can't be detected from planar PIV.

        Procedure: for each phase, the fluctuation u_n' = u_n(phi) - u_n_mean
        is integrated over D and over R in [r, r + width] (R along the
        axis, from the wall).  A sinusoid a sin(phi + phi0) is fitted to the
        8 values, and 2a/tau with tau = 2/f (i.e. a*f) is returned.

        Notes
        -----
        * No density and linear in u_n: the result is a volume-flux-type
          quantity, not a momentum flux (rho u^2), despite the name.
        * Lengths are chord-normalised (x/c): |D| dD dR carries c^3, so
          multiply by c^3 for SI.
        * |D| is computed from unshifted (X, Y).  That is the distance to the
          axis only if the speaker exit point is at the origin; the offset is
          xc * (yc + ry) / r.
        * The phase loop is hard-coded to 8 (np.zeros(8), np.arange(0, 2 pi, pi/4)).

        Parameters
        ----------
        f : float       drive frequency [Hz]
        width : float   integration box size (chord-normalised), both in D and R
        """
        X = self.X
        Y = self.Y
        x = self.x
        y = self.y

        xc = self.PME.runs[self.run]["xc"]
        yc = self.PME.runs[self.run]["yc"]
        r = self.PME.runs[self.run]["r"]

        x_sh = X - xc
        y_sh = Y - yc

        # speak on surface has (unshifted) coordinateds (x=0, r=r)
        rx = -xc
        ry = np.sqrt(r**2 - rx**2)

        # in shifted coordinates, this point is rx, ry
        # in order to integrate a cylindrical shell around this point, it is necessary to calculate the distance
        # from each point to the ray that emanates from the origin and passes through rx, ry (in shifted coords)
        # D = np.abs(ry*X - rx*Y)/r
        # R = np.sqrt(X**2 + Y**2 - D**2)

        theta = np.atan2(ry, rx)

        D = Y*np.cos(theta) - X*np.sin(theta)
        R = X*np.cos(theta) + Y*np.sin(theta)

        # set up R, D grid
        dx = np.abs(x[1] - x[0])
        D_basis = np.arange(-width/2, width/2 + dx, dx)
        R_basis = np.arange(r, r + width + dx, dx)
        D_grid, R_grid = np.meshgrid(D_basis, R_basis)

        x_grid = R_grid*np.cos(theta) - D_grid*np.sin(theta) + xc
        y_grid = R_grid*np.sin(theta) + D_grid*np.cos(theta) + yc

        momentum = np.zeros(8)

        for i in range(self.n_phases):
            u_n = self.fluctuation_phases[i]["u_n"]

            integrand = u_n * np.pi * np.abs(D)
            # plt.pcolormesh(X, Y, integrand)
            # plt.colorbar()
            # plt.show()

            # interpolate onto regularly spaced grid in R (where R is coordinate parallel to ray) and D
            integrand_func_xy = scipy.interpolate.RegularGridInterpolator(
                (x, y), integrand.T,
                method="cubic", fill_value=0, bounds_error=False)

            integrand = integrand_func_xy((x_grid, y_grid))

            # perform double integral over array
            momentum[i] = scipy.integrate.trapezoid(np.trapezoid(integrand, D_grid, axis=1), R_basis, axis=0)

        # print(momentum)
        plt.scatter(np.arange(0, 360, 45), momentum, c='k')
        plt.scatter(np.arange(360, 720, 45), momentum, c='grey')
        # fit sine
        (a, phi), _ = scipy.optimize.curve_fit(lambda x, a, phi: a*np.sin(x + phi), np.arange(0, 2*np.pi, np.pi/4), momentum)
        if a < 0:
            phi += np.pi
            a = np.abs(a)
        plt.plot(np.linspace(0, 720, 1000), a*np.sin(phi + np.deg2rad(np.linspace(0, 720, 1000))))
        plt.xlabel(r"$\phi$ [deg]")
        plt.ylabel(r"$I_j(\phi)$")
        plt.show()
        tau = 2/f

        return 2*a/tau

    def compare_gaussian_models_aic(self, etamin=5e-3, etamax=15e-3, netas=5):
        """STUB: is the dual-Gaussian fit of u_n_mean justified, or overfitting?

        Not implemented yet.  Planned procedure:

        1. For each eta level, sample u_n_mean(theta) along the arc
           r = r_wall + eta (same extraction as fit_dual_gaussians, on the
           cropped self.x, self.y grid).
        2. Fit two nested models with the SAME cubic baseline in theta:
             M1: single Gaussian + cubic   (mu, a, sigma, b, m, q, c)       k = 7
             M2: dual_gaus_third_poly_w_params                            k = 9
           (magnitude_identification uses Gaussian + LINEAR in x, so its fit
           is not directly comparable; don't reuse it here.)
        3. With n points and residual sum of squares RSS,
             AIC  = n ln(RSS/n) + 2k
             AICc = AIC + 2k(k+1)/(n-k-1)
           and report dAIC = AIC_M2 - AIC_M1 per eta. dAIC < -10 is strong
           support for the second Gaussian; |dAIC| < 2 means it isn't justified.
        4. Caveat: neighbouring PIV vectors are correlated (50 % overlap
           windows), so the effective n is smaller than the number of samples
           and AIC will tend to favour M2.  Thin the samples to about one per
           interrogation-window spacing, or treat dAIC as a ranking only.

        Returns (planned)
        -----------------
        dict with etavalues, AIC_single, AIC_dual, dAIC, n_points per level.
        """
        raise NotImplementedError("AIC single- vs dual-Gaussian comparison: stub only")


def dual_gaus_third_poly_w_params(theta, mu, a, fa, sigma, fsigma, b, m, q, c):
    a1 = a
    a2 = -a*fa
    sigma1 = sigma
    sigma2 = sigma*fsigma
    return (
          a1*np.exp(-(theta-mu)**2/(2*sigma1**2))
        + a2*np.exp(-(theta-mu)**2/(2*sigma2**2))
        + b + m*theta + q*theta**2 + c*theta**3
    )


def planar_dual_gauss_third_poly_w_params(x, y, xmu, ymu, a, fa, xs, ys, fxs, fys, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    return (
          a*np.exp(-(x-xmu)**2/(2*xs**2) - (y-ymu)**2/(2*ys**2))
        + fa*np.exp(-(x-xmu)**2/(2*(fxs*xs)**2) - (y-ymu)**2/(2*(fys*ys)**2))
        + np.polynomial.polynomial.polyval2d(x, y, np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8]).reshape(3, 3))
    )


def smallestbox(a):
    r = a.any(1)
    m,n = a.shape
    c = a.any(0)
    out = (slice(r.argmax(), m-r[::-1].argmax()), slice(c.argmax(),n-c[::-1].argmax()))
    return out
