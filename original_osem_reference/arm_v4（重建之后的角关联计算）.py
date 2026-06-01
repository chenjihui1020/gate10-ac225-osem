##重建之后角度计算
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

PI = math.pi

# ===== Geometry (must match CUDA code) =====
scarad = 43.7
absrad = 68.7
rotdir = -1.0


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def position(d: int, ad: int, scaxy: int = 1, absdoi: int = 2, ii: int = 0, jj: int = 0) -> np.ndarray:
    x1 = ad // 8
    z1 = ad - 8 * x1

    x = 11.2 - x1 * 3.2
    y = scarad
    z = 11.2 - z1 * 3.2

    if d < 8:
        xx1 = ii // scaxy
        zz1 = ii - scaxy * xx1
        x = 11.2 - x1 * 3.2 - 1.25 + (2.5 / scaxy) / 2 + (2.5 / scaxy) * xx1
        z = 11.2 - z1 * 3.2 - 1.25 + (2.5 / scaxy) / 2 + (2.5 / scaxy) * zz1

    if d >= 8:
        y = absrad + 9.0 / absdoi / 2.0 + 9.0 / absdoi * jj

    ang = 1.0 * d * rotdir * PI / 4.0
    xx = x * math.cos(ang) + y * math.sin(ang)
    yy = -x * math.sin(ang) + y * math.cos(ang)
    return np.array([xx, yy, z], dtype=float)


def thetaC_from_Es(E_inc: float, Es: float):
    if Es <= 0.0 or Es >= E_inc:
        return None
    costh = 1.0 - 511.0 * (1.0 / (E_inc - Es) - 1.0 / E_inc)
    costh = clamp(costh)
    th = math.acos(costh)
    if math.isnan(th):
        return None
    return th


def voigt_bg(x, amp, mu, sigma, gamma, bg):
    """
    Voigt model with constant background.
    voigt_profile is normalized, so amp is an area-like scale factor.
    """
    return amp * voigt_profile(x - mu, sigma, gamma) + bg


# ================================================================
# User settings
# ================================================================
coin_path = "coin-cc3"
E_inc = 440.0
scalowlim = 20.0
scahighlim = 200.0
sumwin = (0.9, 1.1)
scaxy_setting = 1
absdoi_setting = 2
r0 = np.array([0.0, 0.0, 0.0], dtype=float)

# histogram settings
hist_range = (-60.0, 60.0)
hist_bins = 240

# fit settings after shifting the raw peak to 0 deg
fit_half_window = 10.0
plot_fit_half_window = 15.0
check_windows = [8.0, 10.0, 12.0]

# plotting controls
save_raw_plot = True
save_shifted_plot_no_fit = True
save_shifted_plot_with_fit = True
show_fit_window_lines = True
show_legend = False
# ================================================================


def resolve_coin_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if os.path.exists(p + ".txt"):
        return p + ".txt"
    raise FileNotFoundError(f"Cannot find coin file '{p}' or '{p}.txt' in current directory.")


def interp_x_at_y(x1, y1, x2, y2, y_target):
    if y2 == y1:
        return x1
    return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)


def fwhm_from_hist_local(counts_arr, centers_arr):
    i0 = int(np.argmax(counts_arr))
    half = counts_arr[i0] / 2.0

    iL = i0
    while iL > 0 and counts_arr[iL] >= half:
        iL -= 1

    iR = i0
    while iR < len(counts_arr) - 1 and counts_arr[iR] >= half:
        iR += 1

    if iL == i0 or iR == i0:
        return None

    left = interp_x_at_y(centers_arr[iL], counts_arr[iL],
                         centers_arr[iL + 1], counts_arr[iL + 1], half)
    right = interp_x_at_y(centers_arr[iR - 1], counts_arr[iR - 1],
                          centers_arr[iR], counts_arr[iR], half)
    return right - left


def voigt_fwhm_olivero_longbothum(sigma, gamma):
    """
    Approximate FWHM of Voigt profile:
    F ≈ 0.5346*L + sqrt(0.2166*L^2 + G^2)
    where G = Gaussian FWHM = 2*sqrt(2ln2)*sigma
          L = Lorentzian FWHM = 2*gamma
    """
    sigma = abs(float(sigma))
    gamma = abs(float(gamma))
    G = 2.354820045 * sigma
    L = 2.0 * gamma
    return 0.5346 * L + math.sqrt(0.2166 * L * L + G * G)


def fwhm_from_voigt_model(popt, x_limit=60.0, npts=40001):
    """
    Numerical FWHM from fitted model using:
        half = bg + 0.5 * (peak - bg)
    This is usually better than directly using the analytical approximation
    when a constant background is included.
    """
    amp, mu, sigma, gamma, bg = popt
    f0 = voigt_fwhm_olivero_longbothum(sigma, gamma)
    hw = max(3.0 * f0, 6.0)
    hw = min(hw, x_limit)

    xx = np.linspace(mu - hw, mu + hw, npts)
    yy = voigt_bg(xx, *popt)

    ip = int(np.argmax(yy))
    ypeak = float(yy[ip])
    half = bg + 0.5 * (ypeak - bg)

    iL = ip
    while iL > 0 and yy[iL] >= half:
        iL -= 1

    iR = ip
    while iR < len(yy) - 1 and yy[iR] >= half:
        iR += 1

    if iL == ip or iR == ip or iL == 0 or iR == len(yy) - 1:
        return None

    left = interp_x_at_y(xx[iL], yy[iL], xx[iL + 1], yy[iL + 1], half)
    right = interp_x_at_y(xx[iR - 1], yy[iR - 1], xx[iR], yy[iR], half)
    return right - left


def compute_arms():
    cp = resolve_coin_path(coin_path)
    print(f"Reading events from: {cp}")
    arms = []

    with open(cp, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            d1 = int(parts[0]); ad1 = int(parts[1])
            d2 = int(parts[2]); ad2 = int(parts[3])
            Es = float(parts[4]); Ea = float(parts[5])

            if Es < scalowlim or Es > scahighlim:
                continue

            Etot = Es + Ea
            if Etot < E_inc * sumwin[0] or Etot > E_inc * sumwin[1]:
                continue

            thC = thetaC_from_Es(E_inc, Es)
            if thC is None:
                continue

            r1 = position(d1, ad1, scaxy=scaxy_setting, absdoi=absdoi_setting, ii=0, jj=0)

            for jj in range(absdoi_setting):
                r2 = position(d2, ad2, scaxy=scaxy_setting, absdoi=absdoi_setting, ii=0, jj=jj)

                u = r2 - r1
                s = r1 - r0
                nu = float(np.linalg.norm(u))
                ns = float(np.linalg.norm(s))
                if nu == 0.0 or ns == 0.0:
                    continue

                thG = math.acos(clamp(float(np.dot(u, s) / (nu * ns))))

                # Current convention: ARM = thetaC - thetaG
                # If later you want the other paper's convention, change to:
                # arm_deg = (thG - thC) * 180.0 / PI
                arm_deg = (thC - thG) * 180.0 / PI

                arms.append(arm_deg)

    return np.array(arms, dtype=float)


def fit_core_histogram_voigt(arms_deg: np.ndarray, fit_hw: float):
    counts, edges = np.histogram(arms_deg, bins=hist_bins, range=hist_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mask = (centers >= -fit_hw) & (centers <= fit_hw)
    x = centers[mask]
    y = counts[mask].astype(float)

    positive = y > 0
    x = x[positive]
    y = y[positive]
    if len(x) < 5:
        raise RuntimeError("Too few histogram bins in fit window.")

    bg0 = float(np.percentile(y, 20))
    sigma0 = 2.5
    gamma0 = 2.5
    vp0 = float(voigt_profile(0.0, sigma0, gamma0))
    peak0 = max(float(np.max(y)) - bg0, 1.0)
    amp0 = peak0 / max(vp0, 1e-12)
    mu0 = float(x[np.argmax(y)])

    sigma_y = np.sqrt(y)
    sigma_y[sigma_y < 1.0] = 1.0

    lower = [0.0, -fit_hw, 0.05, 0.05, 0.0]
    upper = [np.inf, fit_hw, 30.0, 30.0, np.inf]

    popt, pcov = curve_fit(
        voigt_bg,
        x,
        y,
        p0=[amp0, mu0, sigma0, gamma0, bg0],
        sigma=sigma_y,
        absolute_sigma=True,
        bounds=(lower, upper),
        maxfev=50000,
    )

    amp, mu, sigma, gamma, bg = popt

    fwhm_approx = voigt_fwhm_olivero_longbothum(sigma, gamma)
    fwhm_numeric = fwhm_from_voigt_model(popt, x_limit=max(abs(hist_range[0]), abs(hist_range[1])))

    return {
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "xfit": x,
        "yfit": y,
        "popt": popt,
        "amp": float(amp),
        "mu": float(mu),
        "sigma": float(abs(sigma)),
        "gamma": float(abs(gamma)),
        "bg": float(bg),
        "fwhm_approx": float(fwhm_approx),
        "fwhm": float(fwhm_numeric if fwhm_numeric is not None else fwhm_approx),
        "pcov": pcov,
    }


def maybe_add_legend():
    if show_legend:
        plt.legend()


def save_raw_plot_func(arms_raw, centers_raw, edges_raw, peak_center_raw):
    plt.figure(figsize=(8, 6))
    counts_raw, _ = np.histogram(arms_raw, bins=hist_bins, range=hist_range)
    width_raw = edges_raw[1] - edges_raw[0]
    plt.bar(centers_raw, counts_raw, width=width_raw, alpha=0.6)
    plt.axvline(peak_center_raw, color="tab:red", linestyle="--", linewidth=1.5)
    plt.title(f"Raw ARM at {E_inc:.0f} keV (N={len(arms_raw)})")
    plt.xlabel(r"ARM ($\theta_C-\theta_G$) [deg]")
    plt.ylabel("Counts")
    plt.xlim(hist_range)
    plt.grid(True, linestyle="--", alpha=0.4)
    maybe_add_legend()
    plt.tight_layout()
    out_raw = f"ARM_{E_inc:.0f}keV_raw_peak.png"
    plt.savefig(out_raw, dpi=300)
    plt.close()
    print(f"Saved figure: {out_raw}")


def save_shifted_plot_no_fit_func(arms_shifted, counts_s, edges_s, centers_s):
    plt.figure(figsize=(8, 6))
    width_s = edges_s[1] - edges_s[0]
    plt.bar(centers_s, counts_s, width=width_s, alpha=0.6)

    if show_fit_window_lines:
        plt.axvline(-fit_half_window, color="tab:red", linestyle=":", linewidth=1.5)
        plt.axvline(+fit_half_window, color="tab:red", linestyle=":", linewidth=1.5)

    plt.title(f"Shifted ARM at {E_inc:.0f} keV (peak shifted to 0°) (N={len(arms_shifted)})")
    plt.xlabel(r"ARM ($\theta_C-\theta_G$) [deg]")
    plt.ylabel("Counts")
    plt.xlim(hist_range)
    plt.grid(True, linestyle="--", alpha=0.4)
    maybe_add_legend()
    plt.tight_layout()
    out_no_fit = f"ARM_{E_inc:.0f}keV_shifted_hist_only.png"
    plt.savefig(out_no_fit, dpi=300)
    plt.close()
    print(f"Saved figure: {out_no_fit}")


def save_shifted_plot_with_fit_func(arms_shifted, counts_s, edges_s, centers_s, fit):
    plt.figure(figsize=(8, 6))
    width_s = edges_s[1] - edges_s[0]
    plt.bar(centers_s, counts_s, width=width_s, alpha=0.6)

    xx = np.linspace(-plot_fit_half_window, plot_fit_half_window, 1200)
    yy = voigt_bg(xx, *fit["popt"])
    plt.plot(xx, yy, linewidth=2.0)

    if show_fit_window_lines:
        plt.axvline(-fit_half_window, color="tab:red", linestyle=":", linewidth=1.5)
        plt.axvline(+fit_half_window, color="tab:red", linestyle=":", linewidth=1.5)

    plt.title(f"Shifted ARM at {E_inc:.0f} keV (peak shifted to 0°) (N={len(arms_shifted)})")
    plt.xlabel(r"ARM ($\theta_C-\theta_G$) [deg]")
    plt.ylabel("Counts")
    plt.xlim(hist_range)
    plt.grid(True, linestyle="--", alpha=0.4)
    maybe_add_legend()
    plt.tight_layout()
    out_with_fit = f"ARM_{E_inc:.0f}keV_shifted_voigtfit.png"
    plt.savefig(out_with_fit, dpi=300)
    plt.close()
    print(f"Saved figure: {out_with_fit}")


def main():
    arms_raw = compute_arms()
    if len(arms_raw) == 0:
        print("No ARM events found after cuts.")
        return

    print("\n=== Raw ARM ===")
    print(f"ARM events (after cuts, incl. DOI integration): {len(arms_raw)}")
    print(f"ARM mean (deg): {np.mean(arms_raw):.6f}")
    print(f"ARM std  (deg): {np.std(arms_raw):.6f}")

    counts_raw, edges_raw = np.histogram(arms_raw, bins=hist_bins, range=hist_range)
    centers_raw = 0.5 * (edges_raw[:-1] + edges_raw[1:])
    i_peak_raw = int(np.argmax(counts_raw))
    peak_center_raw = float(centers_raw[i_peak_raw])
    print(f"Raw histogram peak center (deg): {peak_center_raw:.3f}")

    arms_shifted = arms_raw - peak_center_raw
    print(f"Applied peak shift: ARM_shifted = ARM_raw - ({peak_center_raw:.3f} deg)")

    print("\n=== Shifted ARM ===")
    print(f"ARM mean (deg): {np.mean(arms_shifted):.6f}")
    print(f"ARM std  (deg): {np.std(arms_shifted):.6f}")

    counts_s, edges_s = np.histogram(arms_shifted, bins=hist_bins, range=hist_range)
    centers_s = 0.5 * (edges_s[:-1] + edges_s[1:])
    i_peak_s = int(np.argmax(counts_s))
    peak_center_s = float(centers_s[i_peak_s])
    print(f"Shifted histogram peak center (deg): {peak_center_s:.3f}")

    print("\n=== Core Voigt fit on shifted ARM ===")
    print(f"Fit window used: [-{fit_half_window:.1f}, +{fit_half_window:.1f}] deg")
    fit = fit_core_histogram_voigt(arms_shifted, fit_half_window)

    print(
        f"Voigt fit: mu={fit['mu']:.3f}°, "
        f"sigma(Gauss)={fit['sigma']:.3f}°, "
        f"gamma(Lorentz)={fit['gamma']:.3f}°, "
        f"bg={fit['bg']:.3f}"
    )
    print(f"Voigt FWHM (numeric, recommended): {fit['fwhm']:.3f}°")
    print(f"Voigt FWHM (Olivero-Longbothum approx): {fit['fwhm_approx']:.3f}°")

    fwhm_hist = fwhm_from_hist_local(counts_s, centers_s)
    if fwhm_hist is not None:
        print(f"Histogram FWHM (sanity, full shifted hist): {fwhm_hist:.3f}°")

    print("\n=== Window stability check ===")
    for hw in check_windows:
        try:
            rr = fit_core_histogram_voigt(arms_shifted, hw)
            print(
                f"window ±{hw:>4.1f}° -> "
                f"mu={rr['mu']:>7.3f}°, "
                f"sigma={rr['sigma']:>6.3f}°, "
                f"gamma={rr['gamma']:>6.3f}°, "
                f"FWHM={rr['fwhm']:>6.3f}°"
            )
        except Exception as e:
            print(f"window ±{hw:>4.1f}° -> fit failed: {e}")

    if save_raw_plot:
        save_raw_plot_func(arms_raw, centers_raw, edges_raw, peak_center_raw)
    if save_shifted_plot_no_fit:
        save_shifted_plot_no_fit_func(arms_shifted, counts_s, edges_s, centers_s)
    if save_shifted_plot_with_fit:
        save_shifted_plot_with_fit_func(arms_shifted, counts_s, edges_s, centers_s, fit)


if __name__ == "__main__":
    main()
