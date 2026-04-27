"""
Example — First-order plant with PID controller
G(s) = K / (tau*s + 1)

A first-order system is modelled as a degenerate second-order system with
zeta → ∞, which is equivalent to setting the s² coefficient to zero.
We use scipy.signal directly to keep the transfer function exact.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from pid_simulator import pid_tf, compute_metrics, plot_comparison


# ── First-order plant ─────────────────────────────────────────────────────────

def first_order_plant(K: float, tau: float) -> signal.TransferFunction:
    """G(s) = K / (tau*s + 1)"""
    return signal.TransferFunction([K], [tau, 1])


def closed_loop_fo(plant: signal.TransferFunction,
                   controller: signal.TransferFunction) -> signal.TransferFunction:
    ol_num = np.polymul(controller.num, plant.num)
    ol_den = np.polymul(controller.den, plant.den)
    cl_num = ol_num
    cl_den = np.polyadd(ol_den, ol_num)
    return signal.TransferFunction(cl_num, cl_den)


def simulate_fo(K=1.0, tau=2.0, Kp=2.0, Ki=1.0, Kd=0.0,
                t_end=20.0, dt=0.001, setpoint=1.0):
    G = first_order_plant(K, tau)
    C = pid_tf(Kp, Ki, Kd)
    T = closed_loop_fo(G, C)

    t = np.arange(0, t_end, dt)
    u = np.ones_like(t) * setpoint
    t_out, y_out, _ = signal.lsim(T, u, t)
    metrics = compute_metrics(t_out, y_out, setpoint)
    return t_out, y_out, metrics


# ── Demo: compare P / PI / PID on a first-order plant ─────────────────────────

def main():
    K, tau = 1.0, 2.0
    t_end, sp = 20.0, 1.0

    configs = [
        dict(label="P only   (Kp=3, Ki=0, Kd=0)", Kp=3.0, Ki=0.0, Kd=0.0),
        dict(label="PI       (Kp=3, Ki=1, Kd=0)", Kp=3.0, Ki=1.0, Kd=0.0),
        dict(label="PID      (Kp=3, Ki=1, Kd=0.5)", Kp=3.0, Ki=1.0, Kd=0.5),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(sp, color="gray", linestyle="--", linewidth=1, label="Setpoint")
    band = 0.02 * sp
    ax.axhspan(sp - band, sp + band, alpha=0.10, color="green", label="±2 % band")

    print(f"\nFirst-order plant: K={K}  τ={tau} s\n")
    print(f"{'Controller':<35} {'Tr [s]':>8} {'Ts [s]':>9} {'OS %':>7} {'e_ss':>10}")
    print("─" * 72)

    colors = ["steelblue", "darkorange", "seagreen"]
    for cfg, color in zip(configs, colors):
        Kp, Ki, Kd = cfg["Kp"], cfg["Ki"], cfg["Kd"]
        t, y, m = simulate_fo(K=K, tau=tau, Kp=Kp, Ki=Ki, Kd=Kd,
                               t_end=t_end, setpoint=sp)
        ax.plot(t, y, color=color, linewidth=2, label=cfg["label"])
        print(f"  {cfg['label']:<33} {m['rise_time']:>8.3f} {m['settling_time']:>9.3f}"
              f" {m['overshoot_%']:>7.2f} {m['ss_error']:>10.5f}")

    print("─" * 72)

    ax.set_title(f"First-order plant: K={K}  τ={tau} s — P / PI / PID comparison")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
