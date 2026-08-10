"""Draft figures for Reviewer 2 (b)+(d): frozen-probe SSL-duration + no-SSL baseline.
Internal fold-0 local-scorer values only (EXPERIMENT_RESULTS.md Section 2).

Two SEPARATE figures per plan:
  (a) 224px bulk-only + high-res-tail variants (fig_ssl_duration_224_DRAFTv2)
  (b) 518px full-resolution control, incomplete/stalled run (fig_ssl_duration_fullres_DRAFTv2)
(c) register with/without comparison -- deliberately skipped for now.

x-axis is now a REAL linear epoch count (not categorical) per feedback -- distances
between points are proportional to actual epoch gaps.

Run: python3 plot_ssl_duration.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NOSSL_MRE = 31.27   # probe_nossl -- shared reference point in both figures

# --------------------------------------------------------------------------- #
# (a) 224px bulk + tail variants
# --------------------------------------------------------------------------- #
bulk_x = [0, 10, 20, 60, 100]
bulk_y = [NOSSL_MRE, 26.88, 24.70, 26.43, 25.95]

# (total elapsed epochs, MRE, parent bulk-x it branches from)
tail_points = [
    (65, 25.23, 60),    # probe_dv2_tail_ep60_ep5
    (70, 25.50, 60),    # probe_dv2_tail_ep60_ep10
    (104, 25.09, 100),  # probe_dv2_ep104
]

fig, ax = plt.subplots(figsize=(5.6, 3.6))

ax.plot(bulk_x, bulk_y, "-o", color="#2b6cb0", lw=1.8, ms=5, zorder=4,
        label="224px bulk-only, frozen probe")
ax.plot([bulk_x[0]], [bulk_y[0]], "o", color="#e53e3e", ms=8, zorder=5)
ax.annotate("no SSL\n(off-the-shelf)", (bulk_x[0], bulk_y[0]),
            textcoords="offset points", xytext=(8, 4), fontsize=8, color="#e53e3e")

parent = dict(zip(bulk_x, bulk_y))
for tx, ty, px in tail_points:
    ax.plot([px, tx], [parent[px], ty], "--", color="#38a169", lw=1.1, zorder=2)
ax.plot([t[0] for t in tail_points], [t[1] for t in tail_points], "D",
        color="#38a169", ms=8, zorder=5, label="+ 518px high-res tail")

for x, y in zip(bulk_x, bulk_y):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                fontsize=7.5, ha="center", color="#2b6cb0")
for tx, ty, _ in tail_points:
    ax.annotate(f"{ty:.1f}", (tx, ty), textcoords="offset points", xytext=(6, -11),
                fontsize=7.5, ha="left", color="#38a169")

ax.annotate("+5", (65, 25.23), textcoords="offset points", xytext=(-4, 10), fontsize=6.5, color="#38a169")
ax.annotate("+10", (70, 25.50), textcoords="offset points", xytext=(2, 10), fontsize=6.5, color="#38a169")
ax.annotate("+4", (104, 25.09), textcoords="offset points", xytext=(4, 8), fontsize=6.5, color="#38a169")

ax.set_xlabel("Phase-1 SSL epochs elapsed (cumulative; tail points include a\nshort 518px high-res phase after the labeled 224px bulk stage)")
ax.set_ylabel("Mean radial error, MRE (orig. px) $\\downarrow$")
ax.set_xlim(-4, 112)
ax.set_ylim(22, 33)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_title("(a) 224px bulk-adaptation duration + high-res tail — fold-0 internal", fontsize=9.5)

fig.tight_layout()
fig.savefig("fig_ssl_duration_224_DRAFTv2.pdf")
fig.savefig("fig_ssl_duration_224_DRAFTv2.png", dpi=200)
plt.close(fig)

# --------------------------------------------------------------------------- #
# (b) 518px full-resolution control (ep10/20/30 now all probed)
# --------------------------------------------------------------------------- #
fr_x = [0, 10, 20, 30]
fr_y = [NOSSL_MRE, 25.79, 25.96, 26.92]

fig, ax = plt.subplots(figsize=(5.6, 3.6))

ax.plot(fr_x[1:], fr_y[1:], "-o", color="#805ad5", lw=1.8, ms=5, zorder=4,
        label="518px full-res, frozen probe")
ax.plot([fr_x[0]], [fr_y[0]], "o", color="#e53e3e", ms=8, zorder=5)
ax.annotate("no SSL\n(off-the-shelf)", (fr_x[0], fr_y[0]),
            textcoords="offset points", xytext=(8, 4), fontsize=8, color="#e53e3e")

for x, y in zip(fr_x[1:], fr_y[1:]):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9),
                fontsize=7.5, ha="center", color="#805ad5")

ax.annotate("run evicted mid-ep31 --\nno checkpoint past ep30",
            (20, 23.2), fontsize=7, color="#718096", ha="center", style="italic")

ax.set_xlabel("Phase-1 SSL epochs elapsed (518px throughout, no bulk/tail split)")
ax.set_ylabel("Mean radial error, MRE (orig. px) $\\downarrow$")
ax.set_xlim(-4, 34)
ax.set_ylim(22, 33)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax.set_title("(b) 518px full-resolution control — fold-0 internal", fontsize=9.5)

fig.tight_layout()
fig.savefig("fig_ssl_duration_fullres_DRAFTv2.pdf")
fig.savefig("fig_ssl_duration_fullres_DRAFTv2.png", dpi=200)
plt.close(fig)

print("saved both figures")
