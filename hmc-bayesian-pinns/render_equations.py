"""Render the article's displayed mathematics as Medium-compatible PNGs.

Medium does not reliably render LaTeX source; all strings here are original
equations used in the manuscript and correspond to the documented experiment.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EQUATIONS: dict[str, tuple[str, ...]] = {
    "equation_pinn_loss.png": (
        r"$\mathcal{L}(\theta,\lambda)=w_d\mathcal{L}_{\mathrm{data}}"
        r"+w_f\mathcal{L}_{\mathrm{PDE}}+w_b\mathcal{L}_{\mathrm{BC}}$",
    ),
    "equation_posterior.png": (
        r"$p(\theta,\lambda\mid D)\;\propto\;p(\theta)p(\lambda)"
        r"\exp\!\left[-\frac{1}{2\sigma_d^2}\sum_i(y_i-u_\theta(x_i,t_i))^2"
        r"-\frac{1}{2\sigma_f^2}\sum_j r_{\theta,\lambda}(z_j)^2\right]$",
    ),
    "equation_hamiltonian.png": (
        r"$H(q,p)=U(q)+\frac{1}{2}p^\top M^{-1}p,\quad U(q)=-\log p(q\mid D)$",
        r"$\dot{q}=M^{-1}p,\qquad \dot{p}=-\nabla U(q)$",
    ),
    "equation_metropolis.png": (
        r"$\Delta H = H(q^\prime,p^\prime)-H(q,p),\qquad"
        r"\alpha=\min\!\left(1,\exp(-\Delta H)\right)$",
    ),
    "equation_heat.png": (
        r"$u_t=\kappa u_{xx},\quad u(0,t)=u(1,t)=0,\quad u(x,0)=A\sin(\pi x)$",
        r"$u(x,t;A,\kappa)=A\sin(\pi x)\exp(-\kappa\pi^2 t)$",
    ),
    "equation_sensitivities.png": (
        r"$\alpha=\log A,\quad\beta=\log\kappa$",
        r"$\frac{\partial u}{\partial\alpha}=u,\qquad"
        r"\frac{\partial u}{\partial\beta}=-\kappa\pi^2t\,u$",
    ),
    "equation_fisher.png": (
        r"$I_{\alpha\alpha}=S_0/\sigma_d^2,\quad I_{\alpha\beta}=-cS_1/\sigma_d^2,"
        r"\quad I_{\beta\beta}=c^2S_2/\sigma_d^2$",
        r"$\det I = \frac{c^2}{\sigma_d^4}(S_0S_2-S_1^2),\quad"
        r"S_r=\sum_i u_i^2t_i^r,\quad c=\kappa\pi^2$",
    ),
}


def render(output_dir: Path) -> None:
    """Write one high-resolution transparent-framed raster per equation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, lines in EQUATIONS.items():
        longest = max(len(line) for line in lines)
        width = max(8.2, min(15.0, longest * 0.092))
        height = 0.75 + 0.66 * len(lines)
        fig = plt.figure(figsize=(width, height), facecolor="white")
        for index, equation in enumerate(lines):
            y = (len(lines) - index) / (len(lines) + 1)
            fig.text(0.5, y, equation, ha="center", va="center", fontsize=21, color="#202733")
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight", pad_inches=0.27)
        plt.close(fig)


if __name__ == "__main__":
    render(Path(__file__).parent / "results" / "figures")
