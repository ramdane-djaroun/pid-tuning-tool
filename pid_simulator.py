"""
PID Tuning & Visualization Tool
Second-order plant: G(s) = K / (tau^2 * s^2 + 2*zeta*tau*s + 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# ── Plant & controller parameters ────────────────────────────────────────────

PLANT = {
    "K":    1.0,    # static gain
    "tau":  1.0,    # natural time constant  [s]
    "zeta": 0.3,    # damping ratio (< 1 → underdamped)
}

PID = {
    "Kp": 10.0,
    "Ki": 5.0,
    "Kd": 1.0,
}

SIM = {
    "t_end":    10.0,   # simulation duration [s]
    "dt":       0.001,  # time step [s]
    "setpoint": 1.0,    # step amplitude
}


# ── Transfer function builders ────────────────────────────────────────────────

def plant_tf(K: float, tau: float, zeta: float) -> signal.TransferFunction:
    """G(s) = K / (tau²s² + 2·ζ·τ·s + 1)"""
    num = [K]
    den = [tau**2, 2 * zeta * tau, 1]
    return signal.TransferFunction(num, den)


def pid_tf(Kp: float, Ki: float, Kd: float) -> signal.TransferFunction:
    """C(s) = Kp + Ki/s + Kd·s  →  (Kd·s² + Kp·s + Ki) / s"""
    num = [Kd, Kp, Ki]
    den = [1, 0]
    return signal.TransferFunction(num, den)


def closed_loop(plant: signal.TransferFunction,
                controller: signal.TransferFunction) -> signal.TransferFunction:
    """T(s) = C·G / (1 + C·G)"""
    # multiply numerators and denominators
    ol_num = np.polymul(controller.num, plant.num)
    ol_den = np.polymul(controller.den, plant.den)
    cl_num = ol_num
    cl_den = np.polyadd(ol_den, ol_num)
    return signal.TransferFunction(cl_num, cl_den)


# ── Performance metrics ───────────────────────────────────────────────────────

def compute_metrics(t: np.ndarray, y: np.ndarray, setpoint: float) -> dict:
    sp = setpoint
    error = sp - y

    # Rise time: 10 % → 90 % of setpoint
    try:
        t10 = t[next(i for i, v in enumerate(y) if v >= 0.10 * sp)]
        t90 = t[next(i for i, v in enumerate(y) if v >= 0.90 * sp)]
        rise_time = t90 - t10
    except StopIteration:
        rise_time = float("nan")

    # Settling time (±2 % band)
    band = 0.02 * sp
    settled = np.where(np.abs(error) <= band)[0]
    settling_time = t[settled[-1]] if len(settled) and settled[-1] != len(t) - 1 else float("nan")
    # refine: last crossing out of band
    in_band = np.abs(error) <= band
    out_of_band_after = np.where(~in_band)[0]
    if len(out_of_band_after):
        last_out = out_of_band_after[-1]
        settling_time = t[last_out + 1] if last_out + 1 < len(t) else float("nan")
    else:
        settling_time = t[np.where(in_band)[0][0]] if len(np.where(in_band)[0]) else float("nan")

    # Overshoot
    peak = np.max(y)
    overshoot = 100.0 * (peak - sp) / sp if sp != 0 else 0.0

    # Steady-state error
    ss_error = abs(sp - y[-1])

    return {
        "rise_time":     rise_time,
        "settling_time": settling_time,
        "overshoot_%":   max(overshoot, 0.0),
        "ss_error":      ss_error,
        "peak":          peak,
    }


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(plant_params: dict | None = None,
             pid_params: dict | None = None,
             sim_params: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    p  = {**PLANT, **(plant_params or {})}
    c  = {**PID,   **(pid_params or {})}
    s  = {**SIM,   **(sim_params or {})}

    G   = plant_tf(p["K"], p["tau"], p["zeta"])
    C   = pid_tf(c["Kp"], c["Ki"], c["Kd"])
    T   = closed_loop(G, C)

    t   = np.arange(0, s["t_end"], s["dt"])
    u   = np.ones_like(t) * s["setpoint"]
    t_out, y_out, _ = signal.lsim(T, u, t)

    metrics = compute_metrics(t_out, y_out, s["setpoint"])
    return t_out, y_out, metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_response(t: np.ndarray, y: np.ndarray, metrics: dict,
                  pid_params: dict | None = None,
                  plant_params: dict | None = None,
                  sim_params: dict | None = None,
                  ax: plt.Axes | None = None,
                  label: str | None = None) -> plt.Figure:
    c = {**PID,   **(pid_params or {})}
    p = {**PLANT, **(plant_params or {})}
    s = {**SIM,   **(sim_params or {})}

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    lbl = label or f"Kp={c['Kp']}  Ki={c['Ki']}  Kd={c['Kd']}"
    ax.plot(t, y, linewidth=2, label=lbl)

    if standalone:
        ax.axhline(s["setpoint"], color="gray", linestyle="--", linewidth=1, label="Setpoint")
        band = 0.02 * s["setpoint"]
        ax.axhspan(s["setpoint"] - band, s["setpoint"] + band,
                   alpha=0.12, color="green", label="±2 % band")

        # annotations
        sp = s["setpoint"]
        if not np.isnan(metrics["rise_time"]):
            ax.annotate(
                f"Tr = {metrics['rise_time']:.3f} s",
                xy=(metrics["rise_time"], 0.90 * sp),
                xytext=(metrics["rise_time"] + 0.3, 0.75 * sp),
                arrowprops=dict(arrowstyle="->", color="steelblue"),
                color="steelblue", fontsize=9,
            )
        if not np.isnan(metrics["settling_time"]):
            ax.axvline(metrics["settling_time"], color="orange",
                       linestyle=":", linewidth=1.2, label=f"Ts = {metrics['settling_time']:.3f} s")

        title = (
            f"PID Step Response — 2nd-order plant\n"
            f"K={p['K']}  τ={p['tau']} s  ζ={p['zeta']}  |  "
            f"Kp={c['Kp']}  Ki={c['Ki']}  Kd={c['Kd']}\n"
            f"Tr={metrics['rise_time']:.3f} s   "
            f"Ts={metrics['settling_time']:.3f} s   "
            f"OS={metrics['overshoot_%']:.1f} %   "
            f"e_ss={metrics['ss_error']:.4f}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Output")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

    return fig


def plot_comparison(configs: list[dict]) -> plt.Figure:
    """
    configs: list of dicts, each with optional keys:
        pid_params, plant_params, sim_params, label
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    s_ref = {**SIM, **(configs[0].get("sim_params") or {})}
    ax.axhline(s_ref["setpoint"], color="gray", linestyle="--", linewidth=1, label="Setpoint")
    band = 0.02 * s_ref["setpoint"]
    ax.axhspan(s_ref["setpoint"] - band, s_ref["setpoint"] + band,
               alpha=0.10, color="green")

    for cfg in configs:
        t, y, metrics = simulate(
            cfg.get("plant_params"),
            cfg.get("pid_params"),
            cfg.get("sim_params"),
        )
        plot_response(t, y, metrics,
                      pid_params=cfg.get("pid_params"),
                      sim_params=cfg.get("sim_params"),
                      ax=ax,
                      label=cfg.get("label"))

    ax.set_title("PID Comparison — Step Response")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── CLI entry-point ───────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Interactive PID step-response simulator (2nd-order plant)"
    )
    parser.add_argument("--Kp",   type=float, default=PID["Kp"])
    parser.add_argument("--Ki",   type=float, default=PID["Ki"])
    parser.add_argument("--Kd",   type=float, default=PID["Kd"])
    parser.add_argument("--K",    type=float, default=PLANT["K"],    help="Plant static gain")
    parser.add_argument("--tau",  type=float, default=PLANT["tau"],  help="Plant time constant [s]")
    parser.add_argument("--zeta", type=float, default=PLANT["zeta"], help="Damping ratio")
    parser.add_argument("--tend", type=float, default=SIM["t_end"],  help="Simulation end time [s]")
    parser.add_argument("--sp",   type=float, default=SIM["setpoint"], help="Step setpoint")
    parser.add_argument("--save", type=str,   default=None,          help="Save figure to file")
    return parser.parse_args()


def main():
    args = _parse_args()

    plant_params = {"K": args.K, "tau": args.tau, "zeta": args.zeta}
    pid_params   = {"Kp": args.Kp, "Ki": args.Ki, "Kd": args.Kd}
    sim_params   = {"t_end": args.tend, "setpoint": args.sp}

    t, y, metrics = simulate(plant_params, pid_params, sim_params)

    print("\n-- Simulation Results ------------------------------------------")
    print(f"  Plant   : K={args.K}  tau={args.tau} s  zeta={args.zeta}")
    print(f"  PID     : Kp={args.Kp}  Ki={args.Ki}  Kd={args.Kd}")
    print(f"  Rise time    (10->90 %) : {metrics['rise_time']:.4f} s")
    print(f"  Settling time (+-2 %)   : {metrics['settling_time']:.4f} s")
    print(f"  Overshoot               : {metrics['overshoot_%']:.2f} %")
    print(f"  Steady-state error      : {metrics['ss_error']:.6f}")
    print("----------------------------------------------------------------\n")

    fig = plot_response(t, y, metrics, pid_params, plant_params, sim_params)

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
