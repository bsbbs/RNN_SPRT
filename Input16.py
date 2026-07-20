import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Parameters for the symmetric Gaussians
mu = 2.2   # distance from y=0
sigma = 5.0
x = np.linspace(-15, 15, 500)

# Two symmetric Gaussian PDFs
pdf_left = norm.pdf(x, -mu, sigma)
pdf_right = norm.pdf(x, mu, sigma)

# Log ratio of PDFs
log_ratio = np.log10(pdf_right / pdf_left)

# Define target log ratio values
# target_log_ratios = np.array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# target_log_ratios = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
target_log_ratios = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# target_log_ratios = np.array([-0.3, 0.3])

# Find corresponding x values by interpolation
x_targets = np.interp(target_log_ratios, log_ratio, x)

# Plot with vertical lines at target x values
fig, ax = plt.subplots(3, 1, figsize=(2.5, 3.8), gridspec_kw={'height_ratios': [2, 1, 2]})

pB_raw = norm.pdf(x_targets, mu, sigma)
pA_raw = norm.pdf(x_targets, -mu, sigma)

# Normalize
pB = pB_raw / pB_raw.sum()
pA = pA_raw / pA_raw.sum()

# --- New: set error rates ---
alpha = 0.05   # P_A(x > t_alpha) = α
beta  = 0.05   # P_B(x < t_beta)  = β

# --- Compute thresholds from quantiles ---
t_alpha = norm.ppf(1 - alpha, loc=-mu, scale=sigma)  # right-tail under A
t_beta  = norm.ppf(beta,      loc=+mu, scale=sigma)  # left-tail  under B

# Top plot: PDFs
ax[0].plot(x, pdf_left, label='Target A', color='blue')
ax[0].plot(x, pdf_right, label='Target B', color='red')

# # --- New: shade α and β error areas ---
# # α: under A ~ N(-mu, sigma), right tail x >= t_alpha (blue shade)
# ax[0].fill_between(x, 0, pdf_left, where=(x >= t_alpha), alpha=0.25, color='blue', label=r'$\alpha$ area')

# # β: under B ~ N(+mu, sigma), left tail x <= t_beta (red shade)
# ax[0].fill_between(x, 0, pdf_right, where=(x <= t_beta),  alpha=0.25, color='red',  label=r'$\beta$ area')

# draw decision thresholds as vertical lines (optional but helpful)
# ax[0].axvline(t_alpha, color='blue', linestyle=':', linewidth=1.5)
# ax[0].axvline(t_beta,  color='red',  linestyle=':', linewidth=1.5)

# keep your target x vertical lines
for xt in x_targets:
    ax[0].axvline(xt, color='gray', linestyle='--', alpha=0.7)

ax[0].set_xlim(-12, 12)
ax[0].set_xticks(range(-12, 13, 4))
ax[0].set_ylabel('PDF')
ax[0].legend()
ax[0].set_title('Task Distribution')

# middle plot: Log ratio
ax[1].plot(x, log_ratio, color='green')
for xt in x_targets:
    ax[1].axvline(xt, color='gray', linestyle='--', alpha=0.7)
ax[1].axhline(0, color='black', linestyle='--')
ax[1].set_xlim(-12, 12)
ax[1].set_xticks(range(-12, 13, 4))
# ax[1].set_xlabel('x')
ax[1].set_ylabel('logLR')

# bottom plot: Probability
markerline1, stemlines1, baseline1 = ax[2].stem(x_targets, pA, linefmt='b-', markerfmt='bo', basefmt='k-', label='Target A')
markerline2, stemlines2, baseline2 = ax[2].stem(x_targets, pB, linefmt='r-', markerfmt='ro', basefmt='k-', label='Target B')
markerline1.set_markersize(4)
markerline2.set_markersize(4)
ax[2].legend()
ax[2].set_xlim(-12, 12)
ax[2].set_xticks(range(-12, 13, 4))
# ax[2].set_ylim(0, 0.25)
ax[2].set_xlabel('x')
ax[2].set_ylabel('PMF')

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

# save the figure to svg
plt.savefig('distribution.svg', dpi=300)

plt.show()

