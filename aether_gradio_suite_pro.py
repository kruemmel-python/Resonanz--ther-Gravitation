# -*- coding: utf-8 -*-
"""
Gradio-Suite PRO: Resonanz-Äther Gravitation
Features:
- Tab 1: Fit & Profile (Plots, CSV, PDF, LaTeX-Report)
- Tab 2: Orbit (2D & 3D), Energiecheck, CSV
- Tab 3: Parameter-Scan (RMSE-Linien + Heatmaps)
- Tab 4: Kuppel-Eigenmoden (stehende Radialmoden, Frequenzen, Profile)
- Tab 5: Levitation-Scan (Heatmap L, Time-to-ground-Map)
- Tab 6: Batch-Report (ZIP)
Autor: [Dein Name]
"""

import os, io, zipfile, tempfile, datetime, textwrap, warnings, math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

import gradio as gr

warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

# -------------------- Konstanten --------------------
g0 = 9.81
R  = 6_371_000.0
GM = g0 * R**2
c_a2 = 1.0

def g_target(r): return GM/(r**2)

# -------------------- Modelle --------------------
def g_exp1(r, k):
    A0 = g0/(c_a2*k)
    return c_a2*k*A0*np.exp(-k*(r-R))

def g_exp2(r, k1, k2, w):
    denom = w*k1 + (1-w)*k2
    A = g0/(c_a2*denom)
    return c_a2*A*( w*k1*np.exp(-k1*(r-R)) + (1-w)*k2*np.exp(-k2*(r-R)) )

def g_hybrid(r, pars):
    k1,a1,k2,a2 = pars
    base = GM/(r**2)
    corr = c_a2*(a1*k1*np.exp(-k1*(r-R)) + a2*k2*np.exp(-k2*(r-R)))
    return base + corr

def phi_hybrid(r, pars):
    k1,a1,k2,a2 = pars
    return -GM/r - c_a2*(a1*np.exp(-k1*(r-R)) + a2*np.exp(-k2*(r-R)))

# -------------------- Fit --------------------
def rmse(y, yhat): return float(np.sqrt(np.mean((y-yhat)**2)))

def fit_exp1(r_grid):
    GT = g_target(r_grid)
    def obj(x):
        k = np.exp(x[0])
        return rmse(GT, g_exp1(r_grid, k))
    x0 = np.array([-14.0])
    if SCIPY_OK:
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=8000))
        k = float(np.exp(res.x[0])); f=float(res.fun)
    else:
        best=(None,1e99)
        for _ in range(2500):
            k = np.exp(np.random.uniform(-18,-8))
            f = rmse(GT, g_exp1(r_grid, k))
            if f<best[1]: best=(k,f)
        k,f=best
    return dict(k=k, rmse=f)

def fit_exp2(r_grid):
    GT = g_target(r_grid)
    def obj(x):
        k1=np.exp(x[0]); k2=np.exp(x[1]); w=1/(1+np.exp(-x[2]))
        if k1<=0 or k2<=0: return 1e9
        return rmse(GT, g_exp2(r_grid,k1,k2,w))
    x0 = np.array([-15.0,-10.0,0.0])
    if SCIPY_OK:
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=10000))
        k1=float(np.exp(res.x[0])); k2=float(np.exp(res.x[1])); w=float(1/(1+np.exp(-res.x[2])))
        f=float(res.fun)
    else:
        best=(None,1e99)
        for _ in range(4000):
            k1=np.exp(np.random.uniform(-18,-8))
            k2=np.exp(np.random.uniform(-18,-8))
            w = 1/(1+np.exp(-np.random.normal()))
            f = rmse(GT, g_exp2(r_grid,k1,k2,w))
            if f<best[1]: best=((k1,k2,w),f)
        (k1,k2,w),f=best
    return dict(k1=k1,k2=k2,w=w,rmse=f)

def fit_hybrid(r_grid):
    GT = g_target(r_grid)
    def obj(x):
        k1=np.exp(x[0]); a1=x[1]; k2=np.exp(x[2]); a2=x[3]
        return rmse(GT, g_hybrid(r_grid,[k1,a1,k2,a2]))
    x0 = np.array([-14.0,0.0,-9.0,0.0])
    if SCIPY_OK:
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(maxfev=15000))
        k1=float(np.exp(res.x[0])); a1=float(res.x[1]); k2=float(np.exp(res.x[2])); a2=float(res.x[3])
        f=float(res.fun)
    else:
        best=(None,1e99)
        for _ in range(7000):
            k1=np.exp(np.random.uniform(-18,-8))
            k2=np.exp(np.random.uniform(-18,-8))
            a1=np.random.uniform(-5,5)
            a2=np.random.uniform(-5,5)
            f = rmse(GT, g_hybrid(r_grid,[k1,a1,k2,a2]))
            if f<best[1]: best=([k1,a1,k2,a2],f)
        (k1,a1,k2,a2),f=best
    return dict(k1=k1,a1=a1,k2=k2,a2=a2,rmse=f)

# -------------------- Levitation --------------------
def levitation_factor(fm, fa, bw):
    delta = (fm-fa)/(bw+1e-30)
    return float(np.clip(1.0 - np.exp(-delta**2), 0.0, 1.0))

def simulate_drop_hybrid(pars, h0=10.0, dt=0.002, tmax=6.0, fm=None, fa=None, bw=None):
    n = int(tmax/dt)
    t = np.linspace(0, n*dt, n+1)
    z = np.zeros_like(t); v = np.zeros_like(t)
    z[0]=h0
    for i in range(len(t)-1):
        r = R + max(z[i], 0.0)
        g = g_hybrid(r, pars)
        L = 1.0
        if (fm is not None) and (fa is not None) and (bw is not None):
            L = levitation_factor(fm,fa,bw)
        a = -L * g
        v[i+1] = v[i] + a*dt
        z[i+1] = z[i] + v[i+1]*dt
        if z[i+1] <= 0:
            z[i+1]=0; v[i+1]=0
            t=t[:i+2]; z=z[:i+2]; v=v[:i+2]
            break
    return t,z,v

# -------------------- Plots/Save --------------------
def save_fig(fig, path, dpi=170):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path

def plot_profiles(r_grid, res1, res2, res3, outdir):
    gt = g_target(r_grid)
    g1 = g_exp1(r_grid, res1["k"])
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])

    xkm = (r_grid-R)/1000.0
    fig = plt.figure(figsize=(9,5.4))
    ax = fig.gca()
    ax.plot(xkm, gt, 'k', lw=3, label='Ziel 1/r²', zorder=10)
    ax.plot(xkm, g1, color='#1f77b4', ls='--', lw=2.5, label='A) 1×Exp', zorder=8)
    ax.plot(xkm, g2, color='#ff7f0e', ls=':', lw=3, marker='o', markevery=40, ms=4, label='B) 2×Exp', zorder=12)
    ax.plot(xkm, g3, color='#2ca02c', lw=2.8, alpha=0.8, label='C) Hybrid (best)', zorder=11)
    ax.set_xlabel('Höhe [km]'); ax.set_ylabel('g(r) [m/s²]')
    ax.set_title('Modelle vs. Referenz'); ax.grid(True, alpha=0.3); ax.legend(framealpha=1.0)
    return save_fig(fig, os.path.join(outdir,"profiles.png"))

def plot_residuals(r_grid, res1, res2, res3, outdir):
    gt = g_target(r_grid)
    g1 = g_exp1(r_grid, res1["k"])
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])

    xkm = (r_grid-R)/1000.0
    fig = plt.figure(figsize=(9,4.6)); ax=fig.gca()
    ax.axhline(0,color='k',lw=1)
    ax.plot(xkm, g1-gt, color='#1f77b4', ls='--', lw=2.0, label='A) 1×Exp')
    ax.plot(xkm, g2-gt, color='#ff7f0e', ls=':', lw=2.2, marker='o', markevery=40, ms=3.5, label='B) 2×Exp')
    ax.plot(xkm, g3-gt, color='#2ca02c', lw=2.2, alpha=0.9, label='C) Hybrid')
    ax.set_xlabel('Höhe [km]'); ax.set_ylabel('Δg [m/s²]')
    ax.set_title('Residuen gegenüber 1/r²'); ax.grid(True, alpha=0.3); ax.legend(framealpha=1.0)
    return save_fig(fig, os.path.join(outdir,"residuals.png"))

def plot_percent_error(r_grid, res2, res3, outdir):
    gt = g_target(r_grid)
    g2 = g_exp2(r_grid, res2["k1"],res2["k2"],res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])
    pe2 = (g2-gt)/gt*100.0
    pe3 = (g3-gt)/gt*100.0
    xkm = (r_grid-R)/1000.0
    fig = plt.figure(figsize=(9,4.6)); ax=fig.gca()
    ax.axhline(0,color='black',lw=1)
    ax.plot(xkm, pe2, color='#ff7f0e', ls=':', lw=2.0, label='B) 2×Exp')
    ax.plot(xkm, pe3, color='#2ca02c', lw=2.2, label='C) Hybrid (best)')
    ax.set_xlabel('Höhe [km]'); ax.set_ylabel('Fehler [%]')
    ax.set_title('Relativer Fehler gegenüber 1/r²'); ax.grid(True, alpha=0.3); ax.legend()
    return save_fig(fig, os.path.join(outdir,"percent_error.png"))

def plot_zoom_with_delta(r_grid, res2, res3, outdir):
    gt = g_target(r_grid)
    g2 = g_exp2(r_grid, res2["k1"],res2["k2"],res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])
    xkm = (r_grid-R)/1000.0
    mask = (xkm>=2500) & (xkm<=9000)

    fig, ax = plt.subplots(figsize=(9,5.2), constrained_layout=True)
    ax.plot(xkm[mask], gt[mask], 'k', lw=2.5, label='Ziel 1/r²')
    ax.plot(xkm[mask], g2[mask], color='#ff7f0e', ls=':', lw=2.5, label='B) 2×Exp')
    ax.plot(xkm[mask], g3[mask], color='#2ca02c', lw=2.2, label='C) Hybrid')
    ax.set_xlabel('Höhe [km]'); ax.set_ylabel('g(r) [m/s²]')
    ax.set_title('Zoom: Modelle vs. Referenz (2.5–9 Mm)')
    ax.grid(True, alpha=0.3); ax.legend()

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset = inset_axes(ax, width="45%", height="45%", loc="lower left", borderpad=2)
    inset.axhline(0,color='k',lw=1)
    inset.plot(xkm[mask], 50*(g2[mask]-gt[mask]), color='#ff7f0e', ls=':', lw=2)
    inset.plot(xkm[mask], 50*(g3[mask]-gt[mask]), color='#2ca02c', lw=2)
    inset.set_title('Δg × 50'); inset.grid(True, alpha=0.3)

    return save_fig(fig, os.path.join(outdir,"zoom_delta.png"))

def plot_drop(pars, outdir, h0=10.0, fm=40.0, fa=40.0, bw=5.0):
    t1,z1,v1 = simulate_drop_hybrid(pars, h0=h0)
    t2,z2,v2 = simulate_drop_hybrid(pars, h0=h0, fm=fm, fa=fa, bw=bw)
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(t1,z1,label='Normal'); ax1.plot(t2,z2,'--',label='Resonanz')
    ax1.set_xlabel('Zeit [s]'); ax1.set_ylabel('Höhe [m]'); ax1.grid(); ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(t1,v1); ax2.plot(t2,v2,'--')
    ax2.set_xlabel('Zeit [s]'); ax2.set_ylabel('v [m/s]'); ax2.grid()
    return save_fig(fig, os.path.join(outdir,"drop.png"))

def plot_field_slice(pars, outdir):
    L = 1.2*R; n=120
    x = np.linspace(-L, L, n); y = np.linspace(-L, L, n)
    X,Y = np.meshgrid(x,y); Rxy = np.sqrt(X**2+Y**2)
    Rmin = 0.6*R; Rsafe = np.maximum(Rxy,Rmin)
    Phi = phi_hybrid(Rsafe, pars)
    dr = 1000.0
    dPhidr = (phi_hybrid(Rsafe+dr, pars) - Phi)/dr
    gmag = -dPhidr
    U = gmag*(X/Rsafe); V = gmag*(Y/Rsafe)

    fig, ax = plt.subplots(1,2, figsize=(12,5))
    c0 = ax[0].contourf(X/1000.0, Y/1000.0, Phi, 40, cmap='viridis'); 
    fig.colorbar(c0, ax=ax[0]); ax[0].set_title('Potential Φ')
    try:
        ax[1].streamplot(x/1000.0, y/1000.0, U, V, color='k', density=1.2)
    except Exception:
        ax[1].quiver(X[::3,::3]/1000.0, Y[::3,::3]/1000.0, U[::3,::3], V[::3,::3], color='k', scale=1e7)
    ax[1].set_title('Feldlinien')
    return save_fig(fig, os.path.join(outdir,"field.png"))

# -------------------- CSV & PDF & LaTeX & ZIP --------------------
def make_csv_full(r_grid, res1, res2, res3, outdir):
    gt = g_target(r_grid)
    g1 = g_exp1(r_grid, res1["k"])
    g2 = g_exp2(r_grid, res2["k1"], res2["k2"], res2["w"])
    g3 = g_hybrid(r_grid, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])

    df_full = pd.DataFrame({
        "hoehe_km": (r_grid-R)/1000.0,
        "r_m": r_grid,
        "g_ref": gt,
        "g_exp1": g1,
        "g_exp2": g2,
        "g_hybrid": g3,
        "dg_exp1": g1-gt,
        "dg_exp2": g2-gt,
        "dg_hybrid": g3-gt,
    })
    full_path = os.path.join(outdir, "profile_residuals.csv")
    df_full.to_csv(full_path, index=False)
    return full_path, df_full

def make_csv_table(r_grid, res2, res3, outdir):
    heights_km = np.array([0,1000,3000,6000,9000,12000], dtype=float)
    r_samples = R + heights_km*1000.0
    df_tab = pd.DataFrame({
        "hoehe_km": heights_km,
        "g_ref": g_target(r_samples),
        "g_exp2": g_exp2(r_samples, res2["k1"],res2["k2"],res2["w"]),
        "dg_exp2": g_exp2(r_samples, res2["k1"],res2["k2"],res2["w"])-g_target(r_samples),
        "g_hybrid": g_hybrid(r_samples, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]]),
        "dg_hybrid": g_hybrid(r_samples, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])-g_target(r_samples),
    })
    tab_path = os.path.join(outdir, "summary_table.csv")
    df_tab.to_csv(tab_path, index=False)
    return tab_path, df_tab

def make_pdf(outdir, res1, res2, res3):
    pdf_path = os.path.join(outdir, "report.pdf")
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111); ax.axis('off')
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        text = f"""Resonanz-Äther Gravitation – Report
Zeit: {now}

RMSE:
- 1×Exp: {res1['rmse']:.3e} m/s²
- 2×Exp: {res2['rmse']:.3e} m/s²
- Hybrid: {res3['rmse']:.3e} m/s²

Hybrid erzwingt 1/r²; Exp-Modelle nähern es an."""
        ax.text(0.05, 0.9, "Resonanz-Äther Gravitation", fontsize=18, weight='bold', va='top')
        ax.text(0.05, 0.8, text, fontsize=11, va='top')
        pdf.savefig(fig); plt.close(fig)

        for name in ["profiles.png","residuals.png","percent_error.png","zoom_delta.png","drop.png","field.png"]:
            p = os.path.join(outdir, name)
            if os.path.exists(p):
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_subplot(111); ax.axis('off')
                img = plt.imread(p); ax.imshow(img); ax.set_title(name, fontsize=12)
                pdf.savefig(fig); plt.close(fig)
    return pdf_path

def make_latex_report(outdir, res1, res2, res3):
    tex = r"""\documentclass[a4paper,11pt]{article}
\usepackage{graphicx, amsmath, amssymb, geometry}
\geometry{margin=2cm}
\title{Resonanz-Äther Gravitation -- Kurzreport}
\author{[Dein Name]}
\date{\today}
\begin{document}
\maketitle
\section*{Zusammenfassung}
Wir vergleichen drei Modelle der Gravitation im Rahmen eines Resonanz--Äther-Ansatzes: 1×Exponential, 2×Exponential und ein Hybridmodell, das das Fernfeld $1/r^2$ exakt erzwingt und Nahfeldkorrekturen mittels Exponentials zulässt.

\section*{Ergebnisse}
RMSE (über $r\in[R,\,3R]$):
\begin{itemize}
\item 1×Exp: %.3e m/s$^2$
\item 2×Exp: %.3e m/s$^2$
\item Hybrid: %.3e m/s$^2$
\end{itemize}

\section*{Grafiken}
\begin{center}
\includegraphics[width=0.95\textwidth]{profiles.png}\\
\includegraphics[width=0.95\textwidth]{residuals.png}\\
\includegraphics[width=0.95\textwidth]{percent_error.png}\\
\includegraphics[width=0.95\textwidth]{zoom_delta.png}\\
\includegraphics[width=0.95\textwidth]{drop.png}\\
\includegraphics[width=0.95\textwidth]{field.png}
\end{center}
\end{document}
""" % (res1["rmse"], res2["rmse"], res3["rmse"])
    path = os.path.join(outdir, "report.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    # Hinweis: pdflatex muss extern laufen, wir liefern nur die .tex-Datei
    return path

def zip_outputs(outdir):
    """Packt alle Dateien im outdir in ein ZIP und gibt den Dateipfad (str) zurück."""
    zpath = os.path.join(outdir, "aether_artifacts.zip")
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(outdir):
            z.write(os.path.join(outdir, fn), arcname=fn)
    return zpath


# -------------------- Orbit (2D/3D) --------------------
def g_model_select(r, model, res2, res3):
    if model == "Hybrid":
        return g_hybrid(r, [res3["k1"],res3["a1"],res3["k2"],res3["a2"]])
    elif model == "2×Exp":
        return g_exp2(r, res2["k1"], res2["k2"], res2["w"])
    else:
        return g_target(r)

def simulate_orbit_2d(res2, res3, model="Hybrid",
                      h0_km=400.0, v0=7670.0, fpa_deg=0.0,
                      t_hours=3.0, dt=1.0):
    r0 = R + h0_km*1000.0
    pos = np.array([r0, 0.0], dtype=float)
    angle = np.deg2rad(fpa_deg)
    v_tan = v0*np.cos(angle)
    v_rad = v0*np.sin(angle)
    vel = np.array([v_rad, v_tan], dtype=float)

    n = int(t_hours*3600/dt)
    t = np.linspace(0, n*dt, n+1)
    X = np.zeros((n+1,2)); V = np.zeros((n+1,2)); Rarr = np.zeros(n+1)
    E    = np.zeros(n+1)
    X[0]=pos; V[0]=vel; Rarr[0]=np.linalg.norm(pos)

    def accel(x):
        r = np.linalg.norm(x)
        if r < 0.5*R: r = 0.5*R
        gmag = g_model_select(r, model, res2, res3)
        return -gmag * (x/r)

    a = accel(X[0])
    for i in range(n):
        X[i+1] = X[i] + V[i]*dt + 0.5*a*dt*dt
        a_next = accel(X[i+1])
        V[i+1] = V[i] + 0.5*(a + a_next)*dt
        a = a_next
        Rarr[i+1] = np.linalg.norm(X[i+1])
        Phi = -GM/Rarr[i+1]
        if model=="Hybrid":
            Phi += -c_a2*(res3["a1"]*np.exp(-res3["k1"]*(Rarr[i+1]-R)) + res3["a2"]*np.exp(-res3["k2"]*(Rarr[i+1]-R)))
        E[i+1] = 0.5*np.dot(V[i+1],V[i+1]) + Phi

    df = pd.DataFrame({
        "t_s": t,
        "x_m": X[:,0], "y_m": X[:,1],
        "vx_mps": V[:,0], "vy_mps": V[:,1],
        "r_m": Rarr, "E_spec": E
    })
    return df

def simulate_orbit_3d(res2, res3, model="Hybrid",
                      h0_km=400.0, v0=7670.0, incl_deg=51.6,
                      t_hours=3.0, dt=1.0):
    r0 = R + h0_km*1000.0
    # Start in x–z-Ebene mit Inklination
    inc = np.deg2rad(incl_deg)
    pos = np.array([r0, 0.0, 0.0], dtype=float)
    # Tangentialrichtung senkrecht: in y–z, mit Inklination
    v_dir = np.array([0.0, np.cos(inc), np.sin(inc)], dtype=float)
    vel = v_dir * v0

    n = int(t_hours*3600/dt)
    t = np.linspace(0, n*dt, n+1)
    X = np.zeros((n+1,3)); V = np.zeros((n+1,3)); Rarr = np.zeros(n+1)
    E = np.zeros(n+1)
    X[0]=pos; V[0]=vel; Rarr[0]=np.linalg.norm(pos)

    def accel(x):
        r = np.linalg.norm(x)
        if r < 0.5*R: r = 0.5*R
        gmag = g_model_select(r, model, res2, res3)
        return -gmag * (x/r)

    a = accel(X[0])
    for i in range(n):
        X[i+1] = X[i] + V[i]*dt + 0.5*a*dt*dt
        a_next = accel(X[i+1])
        V[i+1] = V[i] + 0.5*(a + a_next)*dt
        a = a_next
        Rarr[i+1] = np.linalg.norm(X[i+1])
        Phi = -GM/Rarr[i+1]
        if model=="Hybrid":
            Phi += -c_a2*(res3["a1"]*np.exp(-res3["k1"]*(Rarr[i+1]-R)) + res3["a2"]*np.exp(-res3["k2"]*(Rarr[i+1]-R)))
        E[i+1] = 0.5*np.dot(V[i+1],V[i+1]) + Phi

    df = pd.DataFrame({"t_s": t, "x_m": X[:,0], "y_m": X[:,1], "z_m": X[:,2],
                       "vx_mps": V[:,0], "vy_mps": V[:,1], "vz_mps": V[:,2],
                       "r_m": Rarr, "E_spec": E})
    return df

def orbit_plots_2d(df, outdir):
    xkm = df["x_m"]/1000.0; ykm = df["y_m"]/1000.0
    rkm = df["r_m"]/1000.0; Es = df["E_spec"]

    fig = plt.figure(figsize=(6,6)); ax=fig.gca()
    ax.plot(xkm, ykm, lw=2)
    circ = plt.Circle((0,0), R/1000.0, color='C7', alpha=0.3)
    ax.add_artist(circ)
    ax.set_aspect('equal','box'); ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]')
    ax.set_title('Orbit (x–y)')
    p1 = save_fig(fig, os.path.join(outdir,"orbit2d_xy.png"))

    fig = plt.figure(figsize=(11,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(df["t_s"]/3600.0, rkm, lw=2); ax1.set_xlabel('t [h]'); ax1.set_ylabel('r [km]'); ax1.grid(True, alpha=0.3)
    ax1.set_title('Radialabstand')
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(df["t_s"]/3600.0, Es-Es.iloc[0], lw=2)
    ax2.set_xlabel('t [h]'); ax2.set_ylabel('ΔE_spez. [J/kg]'); ax2.grid(True, alpha=0.3)
    ax2.set_title('Energieerhaltung (rel.)')
    p2 = save_fig(fig, os.path.join(outdir,"orbit2d_series.png"))
    return [p1,p2]

def orbit_plots_3d(df, outdir):
    xkm = df["x_m"]/1000.0; ykm = df["y_m"]/1000.0; zkm = df["z_m"]/1000.0
    fig = plt.figure(figsize=(7,6)); ax = fig.add_subplot(111, projection='3d')
    ax.plot(xkm, ykm, zkm, lw=1.8)
    # Kugel (Erde) andeuten
    u = np.linspace(0, 2*np.pi, 40); v = np.linspace(0, np.pi, 25)
    xe = (R/1000.0)*np.outer(np.cos(u), np.sin(v))
    ye = (R/1000.0)*np.outer(np.sin(u), np.sin(v))
    ze = (R/1000.0)*np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xe, ye, ze, color='C7', alpha=0.3, linewidth=0.4)
    ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
    ax.set_title('Orbit 3D')
    ax.view_init(elev=25, azim=45)
    p = save_fig(fig, os.path.join(outdir,"orbit3d.png"))
    return [p]

# -------------------- Kuppel-Eigenmoden (Radial) --------------------
def cavity_eigenmodes(N=5, shell_km=200.0, ca=300000.0):
    """
    Einfache stehende Radialmoden zwischen r=R und r=R+L.
    Dirichlet-Grenzen: Phi(R)=Phi(R+L)=0 => k_n = n*pi/L
    Frequenz f_n = (c_a * k_n) / (2*pi)
    """
    L = shell_km * 1000.0
    if L <= 0: L = 1000.0
    k = np.array([np.pi*n/L for n in range(1, N+1)], dtype=float)
    f = ca * k / (2*np.pi)
    # Radialprofile (normiert): sin(k (r-R))
    r = np.linspace(R, R+L, 400)
    modes = [np.sin(k[n-1]*(r-R)) for n in range(1, N+1)]
    return r, k, f, modes

def plot_eigenmodes(N=5, shell_km=200.0, ca=300000.0, outdir=None):
    r, k, f, modes = cavity_eigenmodes(N, shell_km, ca)
    xkm = (r-R)/1000.0
    fig = plt.figure(figsize=(10,6)); ax=fig.gca()
    for i, m in enumerate(modes, start=1):
        ax.plot(xkm, m/np.max(np.abs(m)), label=f"n={i}, f≈{f[i-1]/1e3:.1f} kHz")
    ax.set_xlabel('Höhe über Boden [km]'); ax.set_ylabel('Mode (norm.)')
    ax.set_title(f'Kuppel-Eigenmoden (L≈{shell_km:.0f} km, c_a≈{ca:.0f} m/s)')
    ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)
    path = os.path.join(outdir, "eigenmodes.png")
    return save_fig(fig, path), pd.DataFrame({"n":np.arange(1,N+1),"k_1_per_m":k,"f_Hz":f})

# -------------------- Levitation-Scan --------------------
def levitation_scan_grid(fm_min=1, fm_max=200, fa_min=1, fa_max=200, bw=5):
    fms = np.arange(fm_min, fm_max+1, 1, dtype=float)
    fas = np.arange(fa_min, fa_max+1, 1, dtype=float)
    Lmap = np.zeros((len(fas), len(fms)))
    for i,fa in enumerate(fas):
        for j,fm in enumerate(fms):
            Lmap[i,j] = levitation_factor(fm, fa, bw)
    return fms, fas, Lmap

def plot_levitation_heatmaps(bw=5, outdir=None, pars_hybrid=None):
    fms, fas, Lmap = levitation_scan_grid(1, 200, 1, 200, bw)
    fig = plt.figure(figsize=(7,5)); ax=fig.gca()
    im = ax.imshow(Lmap, origin='lower', extent=[fms[0], fms[-1], fas[0], fas[-1]], aspect='auto', cmap='magma')
    ax.set_xlabel('fm [Hz]'); ax.set_ylabel('fa [Hz]')
    ax.set_title(f'Levitation-Faktor L (bw={bw} Hz)')
    fig.colorbar(im, ax=ax, label='L')
    path1 = save_fig(fig, os.path.join(outdir,"levitation_L_heatmap.png"))

    # Time-to-ground Map (vereinfachtes Modell): simuliere Drop für einige Punkte diagonal fm≈fa
    if pars_hybrid is None:
        raise ValueError("pars_hybrid benötigt")
    grid = np.linspace(5, 200, 20)
    TT = np.zeros((len(grid), len(grid)))
    for i,fa in enumerate(grid):
        for j,fm in enumerate(grid):
            t,z,v = simulate_drop_hybrid(pars_hybrid, h0=5.0, dt=0.002, tmax=4.0, fm=fm, fa=fa, bw=bw)
            TT[i,j] = t[-1]  # „Zeit bis Boden“ (oder Stopp, wenn schweben: dann nahe tmax)
    fig = plt.figure(figsize=(7,5)); ax=fig.gca()
    im2 = ax.imshow(TT, origin='lower', extent=[grid[0], grid[-1], grid[0], grid[-1]], aspect='auto', cmap='viridis')
    ax.set_xlabel('fm [Hz]'); ax.set_ylabel('fa [Hz]')
    ax.set_title('Time-to-ground [s] (h0=5 m)')
    fig.colorbar(im2, ax=ax, label='s')
    path2 = save_fig(fig, os.path.join(outdir,"levitation_time_heatmap.png"))
    return path1, path2

# -------------------- Pipelines --------------------
def run_fit_pipeline(fitR=3.0, grid_points=600, h0=10.0, fm=40.0, fa=40.0, bw=5.0, make_field=True):
    outdir = tempfile.mkdtemp(prefix="aether_fit_")
    rmin = R; rmax = R*fitR
    r_grid = np.linspace(rmin, rmax, int(grid_points))

    # Fit
    res1 = fit_exp1(r_grid)
    res2 = fit_exp2(r_grid)
    res3 = fit_hybrid(r_grid)

    # Plots
    p_profiles  = plot_profiles(r_grid, res1, res2, res3, outdir)
    p_resid     = plot_residuals(r_grid, res1, res2, res3, outdir)
    p_percent   = plot_percent_error(r_grid, res2, res3, outdir)
    p_zoom      = plot_zoom_with_delta(r_grid, res2, res3, outdir)
    p_drop      = plot_drop([res3["k1"],res3["a1"],res3["k2"],res3["a2"]], outdir, h0=h0, fm=fm, fa=fa, bw=bw)
    p_field     = plot_field_slice([res3["k1"],res3["a1"],res3["k2"],res3["a2"]], outdir) if make_field else None

    # CSV + PDF + TeX
    csv_full, df_full = make_csv_full(r_grid, res1, res2, res3, outdir)
    csv_tab,  df_tab  = make_csv_table(r_grid, res2, res3, outdir)
    pdf_path = make_pdf(outdir, res1, res2, res3)
    tex_path = make_latex_report(outdir, res1, res2, res3)

    imgs = [p_profiles, p_resid, p_percent, p_zoom, p_drop]
    if p_field: imgs.append(p_field)
    files = [csv_full, csv_tab, pdf_path, tex_path]
    return outdir, res1, res2, res3, imgs, df_tab, files

def run_param_scan(R_min=1.5, R_max=4.0, R_step=0.25, npts_list=(300, 600, 900)):
    values_R = np.arange(R_min, R_max+1e-9, R_step)
    results = []
    for mR in values_R:
        r_grid = np.linspace(R, R*mR, 600)
        r2 = fit_exp2(r_grid)["rmse"]
        r3 = fit_hybrid(r_grid)["rmse"]
        results.append((mR, r2, r3))
    df = pd.DataFrame(results, columns=["R_mult","RMSE_2exp","RMSE_hybrid"])

    fig = plt.figure(figsize=(7,5)); ax=fig.gca()
    ax.plot(df["R_mult"], df["RMSE_2exp"], 'o--', label='2×Exp')
    ax.plot(df["R_mult"], df["RMSE_hybrid"], 'o-', label='Hybrid')
    ax.set_xlabel('Fit-Bereich (×R)'); ax.set_ylabel('RMSE [m/s²]'); ax.grid(True, alpha=0.3); ax.legend()
    outdir = tempfile.mkdtemp(prefix="aether_scan_")
    p_line = save_fig(fig, os.path.join(outdir,"scan_lines.png"))

    heat = []
    for mR in values_R:
        for npts in npts_list:
            r_grid = np.linspace(R, R*mR, int(npts))
            e2 = fit_exp2(r_grid)["rmse"]
            e3 = fit_hybrid(r_grid)["rmse"]
            heat.append([mR, npts, e2, e3])
    dfh = pd.DataFrame(heat, columns=["R_mult","npts","RMSE_2exp","RMSE_hybrid"])

    fig = plt.figure(figsize=(7,5)); ax=fig.gca()
    sc = ax.scatter(dfh["R_mult"], dfh["npts"], c=dfh["RMSE_2exp"], cmap='viridis')
    ax.set_xlabel('×R'); ax.set_ylabel('npts'); ax.set_title('RMSE 2×Exp (Farbskala)'); fig.colorbar(sc, ax=ax)
    p_hm2 = save_fig(fig, os.path.join(outdir,"scan_heat_2exp.png"))

    fig = plt.figure(figsize=(7,5)); ax=fig.gca()
    sc = ax.scatter(dfh["R_mult"], dfh["npts"], c=dfh["RMSE_hybrid"], cmap='magma')
    ax.set_xlabel('×R'); ax.set_ylabel('npts'); ax.set_title('RMSE Hybrid (Farbskala)'); fig.colorbar(sc, ax=ax)
    p_hm3 = save_fig(fig, os.path.join(outdir,"scan_heat_hybrid.png"))

    return outdir, df, dfh, [p_line, p_hm2, p_hm3]

# -------------------- Gradio Tabs --------------------
with gr.Blocks(title="Resonanz-Äther Gravitation – PRO Suite") as demo:
    gr.Markdown("# Resonanz-Äther Gravitation – Analytik-Suite (PRO)")

    with gr.Tab("1) Fit & Profile"):
        with gr.Row():
            fitR = gr.Slider(1.5, 4.0, value=3.0, step=0.1, label="Fit-Bereich (×R)")
            npts = gr.Slider(200, 1200, value=600, step=50, label="Rasterpunkte")
            h0   = gr.Slider(1, 50, value=10, step=1, label="Fallhöhe [m]")
            fm   = gr.Slider(1, 200, value=40, step=1, label="Levitation fm [Hz]")
            fa   = gr.Slider(1, 200, value=40, step=1, label="Levitation fa [Hz]")
            bw   = gr.Slider(1, 50, value=5, step=1, label="Levitation Breite [Hz]")
            field_on = gr.Checkbox(True, label="Feld-Plot berechnen")
        btn1 = gr.Button("Fit + Plots + CSV/PDF/TeX")
        gallery1 = gr.Gallery(label="Grafiken", columns=2, height=600)
        table1   = gr.Dataframe(label="Kompakt-Tabelle (Ausgewählte Höhen)")
        files1   = gr.Files(label="Downloads")
        text1    = gr.Markdown()

        def _run_fit(fitR, npts, h0, fm, fa, bw, field_on):
            outdir, res1, res2, res3, imgs, dftab, files = run_fit_pipeline(
                fitR=fitR, grid_points=npts, h0=h0, fm=fm, fa=fa, bw=bw, make_field=field_on
            )
            txt = f"""**RMSE**
- 1×Exp: {res1['rmse']:.3e}
- 2×Exp: {res2['rmse']:.3e}
- Hybrid: {res3['rmse']:.3e}

**Ordner:** {outdir}"""
            return imgs, dftab, files, txt

        btn1.click(_run_fit, [fitR, npts, h0, fm, fa, bw, field_on], [gallery1, table1, files1, text1])

    with gr.Tab("2) Orbit (2D)"):
        with gr.Row():
            model = gr.Dropdown(["Hybrid","2×Exp","1/r²"], value="Hybrid", label="Feldmodell")
            h0o   = gr.Slider(100, 2000, value=400, step=10, label="Start-Höhe [km]")
            v0    = gr.Slider(5000, 12000, value=7670, step=10, label="Start-geschw. [m/s]")
            fpa   = gr.Slider(-30, 30, value=0, step=1, label="Flugpfadwinkel [°]")
            th    = gr.Slider(0.2, 12.0, value=3.0, step=0.2, label="Dauer [h]")
            dt    = gr.Slider(0.2, 10.0, value=1.0, step=0.2, label="dt [s]")
        btn2 = gr.Button("Orbit 2D simulieren")
        gallery2 = gr.Gallery(label="Orbit 2D", columns=2, height=480)
        files2   = gr.Files(label="CSV")
        text2    = gr.Markdown()

        def _run_orbit2(model, h0o, v0, fpa, th, dt):
            outdir, res1, res2, res3, imgs, dftab, files = run_fit_pipeline()
            df = simulate_orbit_2d(res2, res3, model=model, h0_km=h0o, v0=v0, fpa_deg=fpa, t_hours=th, dt=dt)
            csv_path = os.path.join(outdir, "orbit2d.csv"); df.to_csv(csv_path, index=False)
            p_list = orbit_plots_2d(df, outdir)
            files_out = [csv_path]
            txt = f"2D-Orbit: Punkte={len(df)}, ΔE_spez. max ~ {np.max(np.abs(df['E_spec']-df['E_spec'].iloc[0])):.3e} J/kg\nOrdner: {outdir}"
            return p_list, files_out, txt

        btn2.click(_run_orbit2, [model, h0o, v0, fpa, th, dt], [gallery2, files2, text2])

    with gr.Tab("3) Orbit (3D)"):
        with gr.Row():
            model3 = gr.Dropdown(["Hybrid","2×Exp","1/r²"], value="Hybrid", label="Feldmodell")
            h0o3   = gr.Slider(100, 2000, value=400, step=10, label="Start-Höhe [km]")
            v03    = gr.Slider(5000, 12000, value=7670, step=10, label="Start-geschw. [m/s]")
            inc    = gr.Slider(0, 180, value=51.6, step=0.5, label="Inklination [°]")
            th3    = gr.Slider(0.2, 12.0, value=3.0, step=0.2, label="Dauer [h]")
            dt3    = gr.Slider(0.2, 10.0, value=1.0, step=0.2, label="dt [s]")
        btn3 = gr.Button("Orbit 3D simulieren")
        gallery3 = gr.Gallery(label="Orbit 3D", columns=2, height=480)
        files3   = gr.Files(label="CSV")
        text3    = gr.Markdown()

        def _run_orbit3(model3, h0o3, v03, inc, th3, dt3):
            outdir, res1, res2, res3, imgs, dftab, files = run_fit_pipeline()
            df = simulate_orbit_3d(res2, res3, model=model3, h0_km=h0o3, v0=v03, incl_deg=inc, t_hours=th3, dt=dt3)
            csv_path = os.path.join(outdir, "orbit3d.csv"); df.to_csv(csv_path, index=False)
            p_list = orbit_plots_3d(df, outdir)
            files_out = [csv_path]
            txt = f"3D-Orbit: Punkte={len(df)}, ΔE_spez. max ~ {np.max(np.abs(df['E_spec']-df['E_spec'].iloc[0])):.3e} J/kg\nOrdner: {outdir}"
            return p_list, files_out, txt

        btn3.click(_run_orbit3, [model3, h0o3, v03, inc, th3, dt3], [gallery3, files3, text3])

    with gr.Tab("4) Parameter-Scan"):
        with gr.Row():
            rmin_s = gr.Slider(1.5, 3.0, value=1.5, step=0.25, label="×R min")
            rmax_s = gr.Slider(2.0, 4.0, value=4.0, step=0.25, label="×R max")
            rstep  = gr.Slider(0.1, 0.5, value=0.25, step=0.05, label="×R Schritt")
            npts_s = gr.CheckboxGroup(choices=["300","600","900"], value=["300","600","900"], label="Rasterpunkte Heatmap")
        btn4 = gr.Button("Scan starten")
        gallery4 = gr.Gallery(label="Scan", columns=3, height=420)
        table4   = gr.Dataframe(label="Linescan (RMSE vs ×R)")
        text4    = gr.Markdown()

        def _run_scan(rmin_s, rmax_s, rstep, npts_s):
            nlist = tuple(sorted(int(x) for x in npts_s)) if npts_s else (600,)
            outdir, df_line, df_heat, figs = run_param_scan(R_min=rmin_s, R_max=rmax_s, R_step=rstep, npts_list=nlist)
            txt = f"Scan fertig. Zeilen: {len(df_line)} | Heatmap-Punkte: {len(df_heat)} | Ordner: {outdir}"
            return figs, df_line, txt

        btn4.click(_run_scan, [rmin_s, rmax_s, rstep, npts_s], [gallery4, table4, text4])

    with gr.Tab("5) Kuppel-Eigenmoden"):
        with gr.Row():
            Nm   = gr.Slider(1, 12, value=6, step=1, label="Anzahl Moden")
            Lkm  = gr.Slider(10, 500, value=200, step=10, label="Schichtdicke L [km]")
            ca   = gr.Slider(1e3, 5e5, value=3e5, step=1e4, label="Ätherschall c_a [m/s]")
        btn5 = gr.Button("Moden berechnen")
        gallery5 = gr.Gallery(label="Eigenmoden", columns=1, height=400)
        table5   = gr.Dataframe(label="Mode-Tabelle (n, k, f)")
        text5    = gr.Markdown()

        def _run_modes(Nm, Lkm, ca):
            outdir = tempfile.mkdtemp(prefix="aether_modes_")
            p, df = plot_eigenmodes(N=int(Nm), shell_km=float(Lkm), ca=float(ca), outdir=outdir)
            txt = f"Moden berechnet. Ordner: {outdir}"
            return [p], df, txt

        btn5.click(_run_modes, [Nm, Lkm, ca], [gallery5, table5, text5])

    with gr.Tab("6) Levitation-Scan"):
        with gr.Row():
            bwx  = gr.Slider(1, 50, value=5, step=1, label="Resonanzbreite bw [Hz]")
        btn6 = gr.Button("Levitation-Karten erzeugen")
        gallery6 = gr.Gallery(label="Levitation Maps", columns=2, height=420)
        text6    = gr.Markdown()

        def _run_lev(bwx):
            outdir, res1, res2, res3, imgs, dftab, files = run_fit_pipeline()
            p1, p2 = plot_levitation_heatmaps(bw=int(bwx), outdir=outdir,
                                              pars_hybrid=[res3["k1"],res3["a1"],res3["k2"],res3["a2"]])
            txt = f"Levitation-Maps fertig. Ordner: {outdir}"
            return [p1,p2], txt

        btn6.click(_run_lev, [bwx], [gallery6, text6])

    with gr.Tab("7) Batch-Report (ZIP)"):
        gr.Markdown("Erzeugt alle Plots + CSV + PDF + TeX als ZIP.")
        btn7 = gr.Button("Batch ausführen & ZIP bauen")
        files7 = gr.File(label="ZIP Download")

        def _run_zip():
            outdir, res1, res2, res3, imgs, dftab, files = run_fit_pipeline()
            zpath = zip_outputs(outdir)
            return zpath


        btn7.click(_run_zip, None, files7)

if __name__ == "__main__":
    demo.launch(share=True)
