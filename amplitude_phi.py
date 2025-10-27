# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 14:32:33 2025

@author: Enrico
"""

import numpy as np
import cmath
from scipy.special import jv


h = [10e3, 25e3, 50e3, 100e3]   # shell thickness [m]
sigma = [0.5, 1, 1.75]            # conductivity [S/m]
d = 25e3        # ice thickness [m]
R_E = 1561e3    # Europa radius [m]
period = 11.23 * 3600  # synodic period [s]
mu0 = 4 * np.pi * 1e-7

n = 1 # degree  

# Define Bessel function of fractional order
def J(m, z):
    return jv(m, z)


results = []

for hi in h:
    for si in sigma:
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

        # Amplitude and phase
        A = 2 * (r0 / R_E)**3 * abs(Q)
        phi = np.degrees(cmath.phase(Q))  # phase in degrees

        # Store results
        results.append((hi, si, A, phi))

        # Print formatted output
        print(f"h = {hi/1e3:6.1f} km | σ = {si:5.2f} S/m | A = {A:8.4f} | φ = {phi:8.3f}°")


results = np.array(results, dtype=[('h', float), ('sigma', float), ('A', float), ('phi', float)])
