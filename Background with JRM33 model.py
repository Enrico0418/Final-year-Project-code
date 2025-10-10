# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 10:57:36 2025

@author: march
"""

import JupiterMag as jm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from numpy.polynomial import Polynomial
import con2020

flyby_file = input("Enter flyby file").strip()

Europa_list = {"ORB04_EUR_EPHIO.TAB", "ORB11_EUR_EPHIO.TAB", 
               "ORB12_EUR_EPHIO.TAB", "ORB14_EUR_EPHIO.TAB", 
               "ORB15_EUR_EPHIO.TAB", "ORB17_EUR_EPHIO.TAB", 
               "ORB19_EUR_EPHIO.TAB", "ORB25_EUR_EPHIO.TAB", 
               "ORB25_EUR_EPHIO.TAB", "ORB04_EUR_SYS3.TAB",
               "ORB11_EUR_SYS3.TAB", "ORB12_EUR_SYS3.TAB",
               "ORB14_EUR_SYS3.TAB", "ORB15_EUR_SYS3.TAB",
               "ORB19_EUR_SYS3.TAB", "ORB26_EUR_SYS3.TAB"}

Callisto_list = {"ORB03_CALL_CPHIO.TAB", "ORB09_CALL_CPHIO.TAB", 
                 "ORB10_CALL_CPHIO.TAB", "ORB21_CALL_CPHIO.TAB", 
                 "ORB22_CALL_CPHIO.TAB", "ORB23_CALL_CPHIO.TAB", 
                 "ORB30_CALL_CPHIO.TAB", "ORB03_CALL_SYS3.TAB",
                 "ORB09_CALL_SYS3.TAB", "ORB10_CALL_SYS3.TAB",
                 "ORB30_CALL_SYS3.TAB"}

Ganymede_list = {"ORB01_GAN_GPHIO.TAB", "ORB02_GAN_GPHIO.TAB", 
                 "ORB07_GAN_GPHIO.TAB", "ORB08_GAN_GPHIO.TAB", 
                 "ORB09_GAN_GPHIO.TAB", "ORB12_GAN_GPHIO.TAB", 
                 "ORB28_GAN_GPHIO.TAB", "ORB29_GAN_GPHIO.TAB",
                 "ORB01_GAN_SYS3.TAB", "ORB02_GAN_SYS3.TAB",
                 "ORB07_GAN_SYS3.TAB", "ORB08_GAN_SYS3.TAB",
                 "ORB28_GAN_SYS3.TAB", "ORB29_GAN_SYS3.TAB"}

Io_list = {"ORB00_IO_IPHIO.TAB", "ORB24_IO_IPHIO.TAB", 
           "ORB27_IO_IPHIO.TAB", "ORB31_IO_IPHIO.TAB", 
           "ORB32_IO_IPHIO.TAB", "ORB00_IO_SYS3.TAB",
           "ORB24_IO_SYS3.TAB", "ORB27_IO_SYS3.TAB",
           "ORB31_IO_SYS3.TAB", "ORB32_IO_SYS3.TAB"}

base_name = os.path.basename(flyby_file)
name_parts = os.path.splitext(base_name)[0].split('_')

orbit_name = name_parts[0]
moon_name = name_parts[1]

orbit_number = ''.join(filter(str.isdigit, orbit_name))
moon_full_names = {
    "EUR": "Europa",
    "IO": "Io",
    "GAN": "Ganymede",
    "CAL": "Callisto"
}

moon_full = moon_full_names.get(moon_name.upper(), moon_name.capitalize())

# Final formatted title string
title_label = f"Orbit {orbit_number} of {moon_full}"

df = pd.read_csv(flyby_file, delim_whitespace=True, header=None)

# Assign all the columns to a variable
df[0] = df[0].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
df['time'] = df[0].dt.strftime('%H:%M')  # Optional human-readable for table
time = df[0].to_numpy()
BR = df[1].to_numpy()   # Second column for Bx
BT = df[2].to_numpy()   # Third column for By
BP = df[3].to_numpy()   # Fourth column for Bz
BTot = df[4].to_numpy() # Fifth column for B Total
r = df[5].to_numpy()  # 3rd last column for x
theta = df[6].to_numpy()  # 2nd last column for y
phi = df[7].to_numpy()  # Last column for z

# Calculate the distance from the surface of Europa
distance_from_surface = r - 1  # Distance from the surface, Europa centre of mass 0,0,0)

# Convert the time to datetime (assuming it's already in a suitable format)
timestamps = pd.to_datetime(time)

# Convert timestamps to UTC hours (relative to the start time)
start_time = timestamps[0]
time_diff = timestamps - start_time  # This will give a Timedelta
timeUTC = mdates.date2num(df[0])


colat = np.radians(90-theta)
east_long = np.radians(phi)


jm.Internal.Config(Model="jrm33",CartesianIn=False,CartesianOut=False)
Bjr,Bjt,Bjp = jm.Internal.Field(r, colat, east_long)

#be sure to configure external field model prior to tracing
jm.Con2020.Config(equation_type='analytic')
#this may also become necessary with internal models in future, e.g.
#setting the model degree

sph_model = con2020.Model(equation_type='analytic', CartesianIn=False,CartesianOut=False)

# --- PLOTS OF FIT AND DATA (with datetime axis) ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)


Bpol = sph_model.Field(r,colat,east_long)
Bcr = Bpol[:, 0]
Bct = Bpol[:, 1]
Bcp = Bpol[:, 2]

Br_model = Bjr +Bcr
Bt_model = Bjt +Bct
Bp_model = Bjp +Bcp

Bx = BP
By = -BR
Bz = -BT
Bjr_model = Bjr
Bjt_model = Bjt
Bjp_model = Bjp

Br_model *= -1
Bt_model *= -1

Bjr_model *= -1
Bjt_model *= -1

# --- PARAMETERS ---
X_min = 10   # half-window size in minutes (adjust as needed)
Y = 1        # polynomial degree (adjust as needed)

# --- TIME HANDLING ---
# Convert times to seconds relative to start
time_sec = (timestamps - start_time).total_seconds().to_numpy()

# Closest approach (already computed)
closest_idx = np.argmin(distance_from_surface)
t_ca_sec = time_sec[closest_idx]  # seconds from start

# Exclusion window
delta_t = X_min * 60.0  # seconds
t_start = t_ca_sec - delta_t
t_end   = t_ca_sec + delta_t

# Mask for quiet parts (outside exclusion window)
mask = (time_sec < t_start) | (time_sec > t_end)

# --- POLYNOMIAL FITS ---
# Fit each component separately using only quiet parts
coeffs_Bx = Polynomial.fit(time_sec[mask], Bx[mask], deg=Y).convert().coef
coeffs_By = Polynomial.fit(time_sec[mask], By[mask], deg=Y).convert().coef
coeffs_Bz = Polynomial.fit(time_sec[mask], Bz[mask], deg=Y).convert().coef

# Evaluate fitted background over full time axis
Bx_fit = np.polyval(coeffs_Bx[::-1], time_sec)  # reverse for np.polyval
By_fit = np.polyval(coeffs_By[::-1], time_sec)
Bz_fit = np.polyval(coeffs_Bz[::-1], time_sec)
B_fit = np.sqrt(Bx_fit**2 + By_fit**2 + Bz_fit**2)

# Bx
axs[0].plot(timeUTC, Bx, 'm', label='Bx data')
axs[0].plot(timeUTC, Bx_fit, 'r--', label='polinomial Fit ')
axs[0].plot(timeUTC, Bp_model, 'y--', label='JRM33+COS2020')
axs[0].plot(timeUTC, Bjp_model, 'g--', label='JRM33')
axs[0].set_ylabel('Bx (nT)')
axs[0].legend()
axs[0].grid(True)


# By
axs[1].plot(timeUTC, By, 'k', label='By data')
axs[1].plot(timeUTC, By_fit, 'r--', label='polinomial Fit ')
axs[1].plot(timeUTC, Br_model, 'y--', label='JRM33+COS2020')
axs[1].plot(timeUTC, Bjr_model, 'g--', label='JRM33')
axs[1].set_ylabel('By (nT)')
axs[1].legend()
axs[1].grid(True)


# Bz
axs[2].plot(timeUTC, Bz, 'b', label='Bz data')
axs[2].plot(timeUTC, Bz_fit, 'r--', label='polinomial Fit ')
axs[2].plot(timeUTC, Bt_model, 'y--', label='JRM33+COS2020')
axs[2].plot(timeUTC, Bjt_model, 'g--', label='JRM33')
axs[2].set_ylabel('Bz (nT)')
axs[2].set_xlabel('Time (UTC)')
axs[2].legend()
axs[2].grid(True)


# Format x-axis with hours and minutes
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.tick_params(axis='x', rotation=45)


plt.tight_layout()
plt.show()


