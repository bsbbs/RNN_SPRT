import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the symmetric Gaussians
mu = 2.2   # distance from y=0
sigma = 5
x = np.linspace(-16, 16, 1000)

# Two symmetric Gaussian PDFs
pdf_left = norm.pdf(x, -mu, sigma)
pdf_right = norm.pdf(x, mu, sigma)

# Log ratio of PDFs
log_ratio = np.log10(pdf_right / pdf_left)

# Define target log ratio values
target_log_ratios = np.array([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9])
# target_log_ratios = np.linspace(-0.9, 0.9, 8)

# Find corresponding x values by interpolation
x_targets = np.interp(target_log_ratios, log_ratio, x)

# Plot with vertical lines at target x values
fig, ax = plt.subplots(3, 1, figsize=(2.5, 5.2), gridspec_kw={'height_ratios': [3, 1, 3]})

pB_raw = norm.pdf(x_targets, mu, sigma)
pA_raw = norm.pdf(x_targets, -mu, sigma)

# Normalize
pB = pB_raw / pB_raw.sum()
pA = pA_raw / pA_raw.sum()

# Top plot: PDFs
ax[0].plot(x, pdf_left, label='Target A', color='blue')
ax[0].plot(x, pdf_right, label='Target B', color='red')
for xt in x_targets:
    ax[0].axvline(xt, color='gray', linestyle='--', alpha=0.7)
ax[0].set_xlim(-16, 16)
ax[0].set_xlabel('x')
ax[0].set_ylabel('PDF')
ax[0].legend()
ax[0].set_title('Symmetric Gaussian Task Distribution')

# middle plot: Log ratio
ax[1].plot(x, log_ratio, color='green')
for xt in x_targets:
    ax[1].axvline(xt, color='gray', linestyle='--', alpha=0.7)
ax[1].axhline(0, color='black', linestyle='--')
ax[1].set_xlim(-16, 16)
ax[1].set_xlabel('x')
ax[1].set_ylabel('logLR')

# bottom plot: Probability
markerline1, stemlines1, baseline1 = ax[2].stem(x_targets, pA, linefmt='b-', markerfmt='bo', basefmt='k-', label='Target A')
markerline2, stemlines2, baseline2 = ax[2].stem(x_targets, pB, linefmt='r-', markerfmt='ro', basefmt='k-', label='Target B')
markerline1.set_markersize(4)
markerline2.set_markersize(4)
ax[2].legend()
ax[2].set_xlim(-16, 16)
# ax[2].set_ylim(0, 0.25)
ax[2].set_xlabel('x')
ax[2].set_ylabel('Probability')

# hide the xticks
ax[2].set_xticks([])
# hide the xlabel
ax[2].set_xlabel('')

leg2 = ax[2].get_legend()
leg2.get_frame().set_linewidth(0.5)
for lh in leg2.get_lines():
    lh.set_linewidth(1.0)
    lh.set_markersize(3.0)

plt.tight_layout()

# plt.show()

# save the figure to svg
plt.savefig('distribution.svg', dpi=300)

plt.show()