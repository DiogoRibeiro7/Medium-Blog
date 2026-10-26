"""The building blocks: psi and weight functions, and asymptotic efficiencies.

Efficiency of an M-estimator of regression at the normal model:
(E psi'(Z))^2 / E psi(Z)^2, Z ~ N(0, 1). The S-estimator's efficiency is that
of the M-estimator with its own rho (Rousseeuw & Yohai 1984).

Run:  python 00_functions.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

from common import BLUE, FIG, INK2, ORANGE, AQUA, style, write
from rob import B_S, C_HUBER, C_MM, C_S, dpsi_bisq, dpsi_huber, psi_bisq, psi_huber, rho_bisq, w_bisq, w_huber

style()


def eff(psi, dpsi):
    e1 = integrate.quad(lambda z: dpsi(z) * stats.norm.pdf(z), -12, 12, points=[-5, 5], limit=200)[0]
    e2 = integrate.quad(lambda z: psi(z) ** 2 * stats.norm.pdf(z), -12, 12, points=[-5, 5], limit=200)[0]
    return e1 ** 2 / e2


out = []
e_h = eff(lambda z: psi_huber(z, C_HUBER), lambda z: dpsi_huber(z, C_HUBER))
e_b = eff(lambda z: psi_bisq(z, C_MM), lambda z: dpsi_bisq(z, C_MM))
e_s = eff(lambda z: psi_bisq(z, C_S), lambda z: dpsi_bisq(z, C_S))
erho = integrate.quad(lambda z: rho_bisq(z, C_S) * stats.norm.pdf(z), -12, 12)[0]
out.append(f"Asymptotic efficiency at the normal: Huber c = {C_HUBER}: {e_h:.3f}; bisquare c = {C_MM}: "
           f"{e_b:.3f} (MM step); bisquare c = {C_S} (S-estimator, 50% breakdown): {e_s:.3f}; "
           f"E rho_S(Z) = {erho:.4f} (target b = {B_S})")
out.append(f"LAD (median regression) efficiency at the normal: 2/pi = {2 / np.pi:.3f}")
out.append("Breakdown points: OLS 0; LAD, Huber and bisquare M 0 under leverage (1/n); "
           "LTS with h = (n + p + 1)/2, S and MM 50% (asymptotically)")
write("00_functions.txt", "\n".join(out))

u = np.linspace(-8, 8, 801)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
ax = axes[0]
ax.plot(u, u, color=INK2, ls="--", lw=1.6, label="Least squares: ψ(u) = u")
ax.plot(u, psi_huber(u, C_HUBER), color=ORANGE, lw=2, label=f"Huber, c = {C_HUBER}")
ax.plot(u, psi_bisq(u, C_MM), color=BLUE, lw=2, label=f"Bisquare, c = {C_MM:.3f}")
ax.set_ylim(-4, 4)
ax.set_xlabel("Standardised residual u = r / s"); ax.set_ylabel("ψ(u): influence on the fit")
ax.set_title("How much a residual can pull the fit")
ax.legend(fontsize=9, loc="upper left")
ax = axes[1]
ax.plot(u, np.ones_like(u), color=INK2, ls="--", lw=1.6, label="Least squares")
ax.plot(u, w_huber(u, C_HUBER), color=ORANGE, lw=2, label="Huber")
ax.plot(u, w_bisq(u, C_MM), color=BLUE, lw=2, label="Bisquare")
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel("Standardised residual u = r / s"); ax.set_ylabel("Weight w(u) = ψ(u) / u")
ax.set_title("The weight each observation gets in IRLS")
ax.legend(fontsize=9, loc="lower center")
fig.savefig(FIG / "fig2_psi_weights.png"); plt.close(fig)
