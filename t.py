# -*- coding: utf-8 -*-
"""
Resonanz-Äther Gravitation – Bestes Modell + Report + Levitation + Feldvisualisierung
Kompatibel mit Pydroid3 (Smartphone)
Autor: [Dein Name]
Datum: 2025-10-23
"""

import os, textwrap, datetime
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ----------------------------------
# 0) Grundeinstellungen
# ----------------------------------
OUTDIR = "aether_report_out"
os.makedirs(OUTDIR, exist_ok=True)

g0 = 9.81
R = 6_371_000.0
GM = g0 * R**2
c_a2 = 1.0

rmin, rmax = R, 3.0 * R
r_grid = np.linspace(rmin, rmax, 600)
def g_target(r): return GM / (r**2)
GT = g_target(r_grid)

# ----------------------------------
# 1) Modelle
# ----------------------------------
def g_exp1(r, k):
    A0 = g0 / (c_a2 * k)
    return c_a2 * k * A0 * np.exp(-k * (r - R))

def g_exp2(r, k1, k2, w):
    denom = (w * k1 + (1 - w) * k2)
    A = g0 / (c_a2 * denom)
    return c_a2 * A * (w * k1 * np.exp(-k1 * (r - R)) + (1 - w) * k2 * np.exp(-k2 * (r - R)))

def g_hybrid(r, pars):
    k1, a1, k2, a2 = pars
    base = GM / (r**2)
    corr = c_a2 * (a1 * k1 * np.exp(-k1 * (r - R)) + a2 * k2 * np.exp(-k2 * (r - R)))
    return base + corr

def phi_hybrid(r, pars):
    k1, a1, k2, a2 = pars
    return -GM / r - c_a2 * (a1 * np.exp(-k1 * (r - R)) + a2 * np.exp(-k2 * (r - R)))

# ----------------------------------
# 2) Fitting / Optimierung
# ----------------------------------
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))

def fit_exp1():
    def obj(x):
        k = np.exp(x[0])
        return rmse(GT, g_exp1(r_grid, k))
    x0 = np.array([-14.0])
    if SCIPY_OK:
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=8000))
        k = float(np.exp(res.x[0])); f = float(res.fun)
    else:
        best = (None, 1e99)
        for _ in range(3000):
            k = np.exp(np.random.uniform(-18, -8))
            f = rmse(GT, g_exp1(r_grid, k))
            if f < best[1]:
                best = (k, f)
        k, f = best
    return dict(k=k, rmse=f)

def fit_exp2():
    def obj(x):
        k1 = np.exp(x[0])
        k2 = np.exp(x[1])
        w = 1 / (1 + np.exp(-x[2]))
        if k1 <= 0 or k2 <= 0:
            return 1e9
        return rmse(GT, g_exp2(r_grid, k1, k2, w))

    x0 = np.array([-15.0, -10.0, 0.0])
    if SCIPY_OK:
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=10000))
        k1 = float(np.exp(res.x[0]))
        k2 = float(np.exp(res.x[1]))
        w = float(1 / (1 + np.exp(-res.x[2])))
        f = float(res.fun)
    else:
        best = (None, 1e99)
        for _ in range(5000):
            k1 = np.exp(np.random.uniform(-18, -8))
            k2 = np.exp(np.random.uniform(-18, -8))
            w = 1 / (1 + np.exp(-np.random.normal()))
            f = rmse(GT, g_exp2(r_grid, k1, k2, w))
            if f < best[1]:
                best = ((k1, k2, w), f)
        (k1, k2, w), f = best
    return dict(k1=k1, k2=k2, w=w, rmse=f)

def fit_hybrid():
    def obj(x):
        k1 = np.exp(x[0]); a1 = x[1]; k2 = np.exp(x[2]); a2 = x[3]
        g = g_hybrid(r_grid, [k1, a1, k2, a2])
        return rmse(GT, g)
    x0 = np.array([-14.0, 0.0, -9.0, 0.0])
    if SCIPY_OK:
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=15000))
        k1 = float(np.exp(res.x[0])); a1 = float(res.x[1])
        k2 = float(np.exp(res.x[2])); a2 = float(res.x[3])
        f = float(res.fun)
    else:
        best = (None, 1e99)
        for _ in range(8000):
            k1 = np.exp(np.random.uniform(-18, -8))
            k2 = np.exp(np.random.uniform(-18, -8))
            a1 = np.random.uniform(-5, 5)
            a2 = np.random.uniform(-5, 5)
            f = rmse(GT, g_hybrid(r_grid, [k1, a1, k2, a2]))
            if f < best[1]:
                best = ([k1, a1, k2, a2], f)
        (k1, a1, k2, a2), f = best
    return dict(k1=k1, a1=a1, k2=k2, a2=a2, rmse=f)

# ----------------------------------
# 3) Levitation / Resonanz
# ----------------------------------
def levitation_factor(fm, fa, bw):
    delta = (fm - fa) / (bw + 1e-30)
    return float(np.clip(1.0 - np.exp(-delta**2), 0.0, 1.0))

def simulate_drop_hybrid(pars, h0=10.0, dt=0.002, tmax=6.0, fm=None, fa=None, bw=None):
    n = int(tmax / dt)
    t = np.linspace(0, n * dt, n + 1)
    z = np.zeros_like(t); v = np.zeros_like(t)
    z[0] = h0
    for i in range(len(t) - 1):
        r = R + max(z[i], 0.0)
        g = g_hybrid(r, pars)
        L = 1.0
        if (fm is not None) and (fa is not None) and (bw is not None):
            L = levitation_factor(fm, fa, bw)
        a = -L * g
        v[i + 1] = v[i] + a * dt
        z[i + 1] = z[i] + v[i + 1] * dt
        if z[i + 1] <= 0:
            z[i + 1] = 0; v[i + 1] = 0
            t = t[:i + 2]; z = z[:i + 2]; v = v[:i + 2]
            break
    return t, z, v

# ----------------------------------
# 4) Visualisierung – Profile/Residuen/Fehler/Zoom/Feld/Drop
# ----------------------------------
def plot_profiles(res1, res2, res3):
    g1 = g_exp1(r_grid, res1["k"])
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"], res3["a1"], res3["k2"], res3["a2"]])

    xkm = (r_grid - R) / 1000.0
    plt.figure(figsize=(9,5.4))
    plt.plot(xkm, g_target(r_grid), color='k', lw=3.0, label='Ziel 1/r²', zorder=10)
    plt.plot(xkm, g1, color='#1f77b4', lw=2.5, ls='--', alpha=0.95, label='A) 1×Exp', zorder=8)
    plt.plot(xkm, g2, color='#ff7f0e', lw=3.0, ls=':', alpha=0.9, label='B) 2×Exp',
             zorder=12, markevery=40, marker='o', ms=4)
    plt.plot(xkm, g3, color='#2ca02c', lw=2.8, alpha=0.75, label='C) Hybrid (best)', zorder=11)
    plt.xlabel('Höhe [km]'); plt.ylabel('g(r) [m/s²]')
    plt.title('Modelle vs. Referenz'); plt.grid(True, alpha=0.25); plt.legend(framealpha=1.0, fontsize=10)
    path = os.path.join(OUTDIR, "profiles.png")
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()
    return path

def plot_residuals(res1, res2, res3):
    g1 = g_exp1(r_grid, res1["k"])
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"], res3["a1"], res3["k2"], res3["a2"]])
    gt = g_target(r_grid)

    xkm = (r_grid - R) / 1000.0
    plt.figure(figsize=(9,4.6))
    plt.axhline(0, color='k', lw=1)
    plt.plot(xkm, g1-gt, color='#1f77b4', ls='--', lw=2.0, label='A) 1×Exp')
    plt.plot(xkm, g2-gt, color='#ff7f0e', ls=':', lw=2.2, label='B) 2×Exp',
             markevery=40, marker='o', ms=3.5)
    plt.plot(xkm, g3-gt, color='#2ca02c', lw=2.2, alpha=0.9, label='C) Hybrid (best)')
    plt.xlabel('Höhe [km]'); plt.ylabel('Δg [m/s²]')
    plt.title('Residuen gegenüber 1/r²')
    plt.grid(True, alpha=0.25); plt.legend(framealpha=1.0)
    path = os.path.join(OUTDIR, "residuals.png")
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()
    return path

def plot_percent_error(res2, res3):
    gt = g_target(r_grid)
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])

    pe2 = (g2-gt)/gt*100.0
    pe3 = (g3-gt)/gt*100.0

    xkm = (r_grid - R)/1000.0
    plt.figure(figsize=(9,4.6))
    plt.axhline(0, color='k', lw=1)
    plt.plot(xkm, pe2, color='#ff7f0e', ls=':', lw=2.0, label='B) 2×Exp')
    plt.plot(xkm, pe3, color='#2ca02c', lw=2.2, label='C) Hybrid (best)')
    plt.xlabel('Höhe [km]'); plt.ylabel('Fehler [%]')
    plt.title('Relativer Fehler gegenüber 1/r²')
    plt.grid(True, alpha=0.3); plt.legend()
    path = os.path.join(OUTDIR, "percent_error.png")
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()
    return path

def plot_zoom_with_delta(res2, res3):
    """Zoom-Plot + Inset (Δg×50), ohne tight_layout-Warnung."""
    gt = g_target(r_grid)
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])
    xkm = (r_grid - R)/1000.0
    mask = (xkm>=2500) & (xkm<=9000)

    # Wichtig: constrained_layout statt tight_layout
    fig, ax = plt.subplots(figsize=(9,5.2), constrained_layout=True)
    ax.plot(xkm[mask], gt[mask], 'k', lw=2.5, label='Ziel 1/r²')
    ax.plot(xkm[mask], g2[mask], color='#ff7f0e', ls=':', lw=2.5, label='B) 2×Exp')
    ax.plot(xkm[mask], g3[mask], color='#2ca02c', lw=2.2, label='C) Hybrid')
    ax.set_xlabel('Höhe [km]'); ax.set_ylabel('g(r) [m/s²]')
    ax.set_title('Zoom: Modelle vs. Referenz (2.5–9 Mm)')
    ax.grid(True, alpha=0.3); ax.legend()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset = inset_axes(ax, width="45%", height="45%", loc="lower left", borderpad=2)
    inset.axhline(0, color='k', lw=1)
    inset.plot(xkm[mask], 50*(g2[mask]-gt[mask]), color='#ff7f0e', ls=':', lw=2.0, label='50×Δg(B)')
    inset.plot(xkm[mask], 50*(g3[mask]-gt[mask]), color='#2ca02c', lw=2.0, label='50×Δg(C)')
    inset.set_title('Δg × 50'); inset.grid(True, alpha=0.3)

    path = os.path.join(OUTDIR, "zoom_delta.png")
    fig.savefig(path, dpi=170)  # kein tight_layout() hier
    plt.close(fig)
    return path


def plot_drop(pars):
    t1, z1, v1 = simulate_drop_hybrid(pars)
    t2, z2, v2 = simulate_drop_hybrid(pars, fm=40, fa=40, bw=5)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(t1, z1, label='Normal')
    plt.plot(t2, z2, '--', label='Resonanz')
    plt.xlabel('Zeit [s]'); plt.ylabel('Höhe [m]')
    plt.grid(); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(t1, v1); plt.plot(t2, v2, '--')
    plt.xlabel('Zeit [s]'); plt.ylabel('v [m/s]')
    plt.grid()
    path = os.path.join(OUTDIR, "drop.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

def plot_field_slice(pars):
    L = 1.2 * R
    n = 100
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y)
    Rxy = np.sqrt(X**2 + Y**2)
    Rmin = 0.6 * R
    Rsafe = np.maximum(Rxy, Rmin)
    Phi = phi_hybrid(Rsafe, pars)
    dr = 1000.0
    dPhidr = (phi_hybrid(Rsafe + dr, pars) - Phi) / dr
    gmag = -dPhidr
    U = gmag * (X / Rsafe)
    V = gmag * (Y / Rsafe)
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    c0 = ax[0].contourf(X/1000, Y/1000, Phi, 40, cmap='viridis')
    fig.colorbar(c0, ax=ax[0])
    ax[0].set_title('Potential Φ')
    try:
        ax[1].streamplot(x/1000, y/1000, U, V, color='k', density=1.2)
    except Exception:
        ax[1].quiver(X[::3,::3]/1000, Y[::3,::3]/1000, U[::3,::3], V[::3,::3], color='k', scale=1e7)
    ax[1].set_title('Feldlinien')
    path = os.path.join(OUTDIR, "field.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
    return path

# ----------------------------------
# 5) Numerische Diagnostik & Report
# ----------------------------------
def diagnostics_print(res2, res3):
    """Numerische Checks: Fehlerstatistiken & Tabelle ausgewählter Höhen."""
    heights_km = np.array([0, 1000, 3000, 6000, 9000, 12000], dtype=float)
    r_samples = R + heights_km*1000.0
    gt = g_target(r_samples)

    g2 = g_exp2(r_samples, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_samples, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])

    err2 = g2 - gt
    err3 = g3 - gt
    rel2 = err2/gt*100.0
    rel3 = err3/gt*100.0

    print("\n--- Detaildiagnose vs. 1/r^2 ---")
    print("Höhe[km]   g_Ziel   g_B(2×Exp)  Δg_B   Rel_B[%]   g_C(Hybrid)  Δg_C   Rel_C[%]")
    for hk, gz, gb, e2, r2, gc, e3, r3 in zip(heights_km, gt, g2, err2, rel2, g3, err3, rel3):
        print(f"{hk:7.0f}  {gz:7.3f}  {gb:10.3f}  {e2:+6.3f}  {r2:+7.3f}   {gc:10.3f}  {e3:+6.3f}  {r3:+7.4f}")

    g2_full = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3_full = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])
    err2_full = g2_full - g_target(r_grid)
    err3_full = g3_full - g_target(r_grid)

    print("\nMax |Δg| B (2×Exp): {:.4f} m/s²".format(np.max(np.abs(err2_full))))
    print("Max |Δg| C (Hybrid): {:.4e} m/s²".format(np.max(np.abs(err3_full))))
    print("Max Rel-Fehler B: {:.3f}%".format(np.max(np.abs(err2_full/g_target(r_grid)*100.0))))

def save_report(res1, res2, res3, figs):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    rel = lambda f: f / g0 * 100
    best_rmse = min(res1["rmse"], res2["rmse"], res3["rmse"])
    best_name = "Hybrid" if best_rmse == res3["rmse"] else ("2×Exp" if best_rmse == res2["rmse"] else "1×Exp")
    verdict = "OK" if rel(best_rmse) < 2 else "NICHT OK"

    md = f"""# Resonanz-Äther Gravitation – Report
**Zeit:** {now}  
**Bestes Modell:** {best_name}  
**RMSE:** {best_rmse:.3e} m/s² ({rel(best_rmse):.3f}% Abweichung)  
**Bewertung:** {verdict}

## Abbildungen
![]({os.path.basename(figs['profiles'])})
![]({os.path.basename(figs['residuals'])})
![]({os.path.basename(figs['percent'])})
![]({os.path.basename(figs['zoom'])})
![]({os.path.basename(figs['drop'])})
![]({os.path.basename(figs['field'])})
"""
    path = os.path.join(OUTDIR, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(md))
    return path

# ----------------------------------
# 6) Hauptprogramm
# ----------------------------------
def main():
    print("Fitting ...")
    res1 = fit_exp1()
    res2 = fit_exp2()
    res3 = fit_hybrid()
    print("Fertig.\nErstelle Diagramme ...")

    f1  = plot_profiles(res1, res2, res3)
    f1b = plot_residuals(res1, res2, res3)
    fpe = plot_percent_error(res2, res3)
    fzm = plot_zoom_with_delta(res2, res3)
    f2  = plot_drop([res3["k1"], res3["a1"], res3["k2"], res3["a2"]])
    f3  = plot_field_slice([res3["k1"], res3["a1"], res3["k2"], res3["a2"]])

    diagnostics_print(res2, res3)

    rep = save_report(
        res1, res2, res3,
        {"profiles": f1, "residuals": f1b, "percent": fpe, "zoom": fzm, "drop": f2, "field": f3}
    )

    print("=== Ergebnisse ===")
    print(f"1×Exp  RMSE = {res1['rmse']:.3e}")
    print(f"2×Exp  RMSE = {res2['rmse']:.3e}")
    print(f"Hybrid RMSE = {res3['rmse']:.3e} (beste Anpassung)")
    print(f"Report & Bilder: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
