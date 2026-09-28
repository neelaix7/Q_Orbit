import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.facecolor"] = "#04060f"
plt.rcParams["axes.facecolor"] = "#04060f"

rng = np.random.default_rng(42)

fig = plt.figure(figsize=(12, 6.75), dpi=150)
ax = fig.add_subplot(111)
ax.set_facecolor("#04060f")
fig.patch.set_facecolor("#04060f")

# faint starfield
for _ in range(400):
    x = rng.uniform(0, 1)
    y = rng.uniform(0, 1)
    s = rng.uniform(0.3, 1.6)
    a = rng.uniform(0.2, 0.9)
    ax.scatter(x, y, s=s, c="white", alpha=a, linewidths=0)

# entangled qubit orbitals - 8 glowing loops
n_q = 8
theta = np.linspace(0, 2 * np.pi, 400)
colors = plt.cm.viridis(np.linspace(0, 0.85, n_q))
for i in range(n_q):
    x0 = (i + 0.5) / n_q
    y0 = 0.5
    r = 0.045 + 0.012 * np.sin(theta * 2 + i)
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta) * 0.5 + 0.08 * np.sin(theta + i * 1.7)
    ax.plot(x, y, color=colors[i], lw=1.4, alpha=0.9)
    ax.scatter([x0], [y0], s=90, color=colors[i], edgecolor="white", linewidth=0.6, zorder=5)

# connecting entangling lines
for i in range(n_q - 1):
    ax.plot([(i + 0.5) / n_q + 0.05, (i + 1.5) / n_q - 0.05],
            [0.5, 0.5], color="#4fc3f7", alpha=0.25, lw=1, ls=":")

# bra-ket labels
for i in range(n_q):
    ax.text((i + 0.5) / n_q, 0.62, f"|{i}>", ha="center", va="bottom",
            color=colors[i], fontsize=11, fontfamily="monospace")

ax.text(0.5, 0.92, "QUANTUM SUPERPOSITION", ha="center", va="center",
        color="#a5d8ff", fontsize=20, fontweight="bold", fontfamily="serif")
ax.text(0.5, 0.85, "8 entangled qubits | default.qubit simulator", ha="center",
        va="center", color="#9fb3d8", fontsize=12)
ax.text(0.5, 0.14, "\u2220 \u03c8 \u27e9 = \u03a3 c_k \u2223 k \u27e9", ha="center",
        va="center", color="#c084fc", fontsize=16, fontfamily="serif")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.tight_layout()
out = r"D:\Capstone\assets\images\quantum_visual.jpg"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("SAVED", out)