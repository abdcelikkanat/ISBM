import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta


fig, axes = plt.subplots(4, 3, figsize=(15, 18))
fig.suptitle(
    "IRM Generative Model Parameters: Beta Bias & CRP Concentration",
    fontsize=20,
    y=0.98,
)


ab_pairs = [
    (0.5, 0.5),
    (1, 1),
    (5, 5),
    (5, 1),
    (10, 2),
    (20, 2),
    (1, 5),
    (2, 10),
    (2, 20),
]

x = np.linspace(0, 1, 500)

for i, (a, b) in enumerate(ab_pairs):
    row = i // 3
    col = i % 3
    ax = axes[row, col]

    y = beta.pdf(x, a, b)
    ax.plot(x, y, color="blue", lw=2)
    ax.fill_between(x, 0, y, alpha=0.2, color="blue")

    # Formatting the subplot
    ax.set_title(f"a={a}, b={b}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(y) * 1.1 if max(y) != np.inf else 5)
    ax.set_yticks([])

    # Contextual labels for IRM
    if a > b:
        desc = "High Link Probability\n(Clusters strongly connect)"
    elif b > a:
        desc = "Low Link Probability\n(Clusters rarely connect)"
    elif a < 1 and b < 1:
        desc = "Polarized\n(Either strongly connect OR don't)"
    else:
        desc = "Uniform/Symmetric\n(Uncertain connection)"

    ax.text(
        0.5,
        ax.get_ylim()[1] * 0.8,
        desc,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    if row == 2:
        ax.set_xlabel("Probability of Link (Theta)", fontsize=12)


alphas = [0.5, 2.0, 10.0]
N_objects = np.arange(1, 51)

for col in range(3):
    ax = axes[3, col]
    alpha_val = alphas[col]

    prob_new_cluster = alpha_val / (N_objects - 1 + alpha_val)

    ax.plot(N_objects, prob_new_cluster, color="red", lw=2)
    ax.fill_between(N_objects, 0, prob_new_cluster, alpha=0.2, color="red")

    ax.set_title(
        f"CRP with Concentration alpha={alpha_val}",
        fontsize=14,
        fontweight="bold",
        color="darkred",
    )
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of Objects Processed", fontsize=12)

    if col == 0:
        ax.set_ylabel("Prob. of Creating\nNEW Cluster", fontsize=12)

    # Contextual text
    if alpha_val < 1:
        desc = "Few Large Clusters\n(Reluctant to make new ones)"
    elif alpha_val > 5:
        desc = "Many Small Clusters\n(Eager to make new ones)"
    else:
        desc = "Moderate Clustering"

    ax.text(
        25,
        0.8,
        desc,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"),
    )

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig("IRM_Generative_Distributions.png", dpi=300, bbox_inches="tight")
print("Figure successfully saved as 'IRM_Generative_Distributions.png'")
plt.show()
