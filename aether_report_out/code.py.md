# -*- coding: utf-8 -*-
"""
Resonanz-Äther Gravitation – Bestes Modell + Report + Levitation + Feldvisualisierung
Kompatibel mit Pydroid3 (Smartphone)
Autor: [Dein Name]
Datum: 2025-10-23
"""

import os, math, textwrap, datetime
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
# 4) Visualisierung
# ----------------------------------
def plot_profiles(res1, res2, res3):
    g1 = g_exp1(r_grid, res1["k"])
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"], res3["a1"], res3["k2"], res3["a2"]])

    plt.figure(figsize=(8,5))
    plt.plot((r_grid - R)/1000, GT, 'k', lw=2, label='Ziel 1/r²')
    plt.plot((r_grid - R)/1000, g1, '--', label='A) 1×Exp')
    plt.plot((r_grid - R)/1000, g2, ':', label='B) 2×Exp')
    plt.plot((r_grid - R)/1000, g3, '-', label='C) Hybrid (best)')
    plt.xlabel('Höhe [km]'); plt.ylabel('g(r) [m/s²]')
    plt.title('Modelle vs. Referenz'); plt.grid(); plt.legend()
    path = os.path.join(OUTDIR, "profiles.png")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()
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
# 5) Report
# ----------------------------------
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
    f1 = plot_profiles(res1, res2, res3)
    f2 = plot_drop([res3["k1"], res3["a1"], res3["k2"], res3["a2"]])
    f3 = plot_field_slice([res3["k1"], res3["a1"], res3["k2"], res3["a2"]])
    rep = save_report(res1, res2, res3, {"profiles": f1, "drop": f2, "field": f3})
    print("=== Ergebnisse ===")
    print(f"1×Exp  RMSE = {res1['rmse']:.3e}")
    print(f"2×Exp  RMSE = {res2['rmse']:.3e}")
    print(f"Hybrid RMSE = {res3['rmse']:.3e} (beste Anpassung)")
    print(f"Report & Bilder: {os.path.abspath(OUTDIR)}")

if __name__ == "__main__":
    main()
