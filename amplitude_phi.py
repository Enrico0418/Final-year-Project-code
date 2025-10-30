# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 14:32:33 2025

@author: Enrico
"""

import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.special import jv

# --- Constants ---
d = 25e3        # ice thickness [m]
R_E = 1561e3    # Europa radius [m]
period = 11.23 * 3600  # synodic period [s]
mu0 = 4 * np.pi * 1e-7
n = 1  # spherical harmonic degree 
omega = 2*np.pi/period 

# --- Helper function ---
def J(m, z):
    return jv(m, z)

# --- Parameter ranges ---
h_vals = np.linspace(5e3, 200e3, 80)      # shell thickness [m]
sigma_vals = np.linspace(0.1, 1, 80)    # conductivity [S/m]

# Prepare grids (σ on x-axis, h on y-axis)
S, H = np.meshgrid(sigma_vals, h_vals)
A = np.zeros_like(S)
phi = np.zeros_like(S)

# --- Compute amplitude and phase ---
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        hi = H[i, j]
        si = S[i, j]

        # Complex wavenumber
        k = (1 + 1j) * np.sqrt(np.pi * mu0 * si / period)

        # Radii
        r0 = R_E - d
        r1 = R_E - d - hi
        r1k = r1 * k
        r0k = r0 * k

        # xi ratio
        xi = (r1k * J(-n - 3/2, r1k)) / ((2*n + 1)*J(n + 1/2, r1k) - r1k * J(n - 1/2, r1k))

        # Q ratio
        Q = - (n * (xi * J(n + 3/2, r0k) - J(-n - 3/2, r0k))) / ((n + 1)*(xi * J(n - 1/2, r0k) - J(-n + 1/2, r0k)))

        # Amplitude and phase (absolute value of phase)
        A[i, j] = 2 * (r0 / R_E)**3 * abs(Q)
        phi[i, j] = abs(np.degrees(cmath.phase(Q)))

# --- Contour level definitions ---
A_levels = np.arange(0.1, 1.0, 0.1)
phi_levels = np.arange(10, 90, 10)
gray_mask = (S * (H/1e3)) >= 50  # since h is plotted in km

# --- Plot 1: Amplitude contours ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
cont_A = ax1.contour(S, H/1e3, A, levels=A_levels, colors='blue', linewidths=1)
ax1.clabel(cont_A, inline=True, fontsize=8, fmt="%.1f")
ax1.contourf(S, H/1e3, gray_mask, levels=[0.5, 1], colors='gray', alpha=0.3)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel("Conductivity σ [S/m]")
ax1.set_ylabel( "Thickness h [km]")
ax1.set_title("A")

# --- Plot 2: Phase contours (absolute value) ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
cont_phi = ax2.contour(S, H/1e3, phi, levels=phi_levels, colors='red', linewidths=1)
ax2.clabel(cont_phi, inline=True, fontsize=8, fmt="%.0f°")
ax2.contourf(S, H/1e3, gray_mask, levels=[0.5, 1], colors='gray', alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Conductivity σ [S/m]")
ax2.set_ylabel("Thickness h [km]")
ax2.set_title("φ contours")

plt.tight_layout()
plt.show()

sigma_vals = np.linspace(0.1, 15, 25)
skin_depth_val = np.sqrt(2/(sigma_vals*mu0*omega))
for i in range(0, len(sigma_vals)):
    print(sigma_vals[i], skin_depth_val[i]/1000)
