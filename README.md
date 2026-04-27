# PID Tuning Tool

Interactive Python tool for simulating, tuning, and visualising PID controllers on continuous-time plants.

---

## Features

- **Second-order plant simulation** — configurable gain K, time constant τ, damping ratio ζ
- **PID controller** — Kp, Ki, Kd tunable from the command line or programmatically
- **Automatic performance metrics** — rise time (10→90 %), settling time (±2 %), overshoot, steady-state error
- **Multi-configuration comparison plots** — overlay several tunings on one figure
- **First-order example** — demonstrates P / PI / PID comparison on a first-order plant

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd pid-tuning-tool

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, numpy, scipy, matplotlib.

---

## Usage

### Command-line

```bash
# Default parameters (second-order plant, PID)
python pid_simulator.py

# Custom PID gains
python pid_simulator.py --Kp 15 --Ki 8 --Kd 2

# Custom plant + controller + simulation horizon
python pid_simulator.py --K 1 --tau 0.5 --zeta 0.2 --Kp 20 --Ki 10 --Kd 3 --tend 5

# Save figure instead of displaying it
python pid_simulator.py --Kp 10 --Ki 5 --Kd 1 --save response.png
```

All CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--Kp` | 10.0 | Proportional gain |
| `--Ki` | 5.0 | Integral gain |
| `--Kd` | 1.0 | Derivative gain |
| `--K` | 1.0 | Plant static gain |
| `--tau` | 1.0 | Plant natural time constant [s] |
| `--zeta` | 0.3 | Damping ratio |
| `--tend` | 10.0 | Simulation end time [s] |
| `--sp` | 1.0 | Step setpoint amplitude |
| `--save` | — | Save figure to file path |

### Python API

```python
from pid_simulator import simulate, plot_response, plot_comparison

# Single simulation
t, y, metrics = simulate(
    plant_params={"K": 1.0, "tau": 1.0, "zeta": 0.3},
    pid_params={"Kp": 10.0, "Ki": 5.0, "Kd": 1.0},
    sim_params={"t_end": 10.0, "setpoint": 1.0},
)
print(metrics)
# {'rise_time': ..., 'settling_time': ..., 'overshoot_%': ..., 'ss_error': ...}

fig = plot_response(t, y, metrics)
fig.savefig("response.png", dpi=150)

# Multi-configuration comparison
fig = plot_comparison([
    {"pid_params": {"Kp": 5,  "Ki": 2, "Kd": 0.5}, "label": "Conservative"},
    {"pid_params": {"Kp": 10, "Ki": 5, "Kd": 1.0}, "label": "Moderate"},
    {"pid_params": {"Kp": 20, "Ki": 10, "Kd": 2.0}, "label": "Aggressive"},
])
fig.show()
```

---

## Examples

### First-order plant — P / PI / PID comparison

```bash
python examples/example_first_order.py
```

Output (console):

```
First-order plant: K=1.0  τ=2.0 s

Controller                          Tr [s]    Ts [s]    OS %       e_ss
────────────────────────────────────────────────────────────────────────
  P only   (Kp=3, Ki=0, Kd=0)       0.537    17.xxx    0.00    0.25000
  PI       (Kp=3, Ki=1, Kd=0)       0.583     4.xxx    3.xx    0.00000
  PID      (Kp=3, Ki=1, Kd=0.5)     0.xxx     2.xxx    x.xx    0.00000
```

The comparison plot shows how adding integral action eliminates steady-state error and derivative action reduces overshoot.

---

## Plant model

```
         K
G(s) = ──────────────────────────────
        τ²s² + 2ζτs + 1
```

Closed-loop with unity-feedback PID:

```
         C(s)·G(s)          (Kd·s² + Kp·s + Ki)·K
T(s) = ─────────────  ,  C(s) = ─────────────────────
        1 + C(s)·G(s)                   s
```

---

## Project structure

```
pid-tuning-tool/
├── pid_simulator.py          # Main simulator, CLI entry-point
├── requirements.txt
├── README.md
└── examples/
    └── example_first_order.py
```

---

## Author

Ramdane Djaroun — Master Automatique 2024
