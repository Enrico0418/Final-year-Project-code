# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 17:54:41 2025

@author: Enrico
"""


import JupiterMag as jm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import matplotlib.dates as mdates
from numpy.polynomial import Polynomial
import con2020

flyby_file = input("Enter SYS3 flyby file").strip()
background_file = input("Enter corresponding moon-centered flyby file").strip()

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

R_moon = 0

if flyby_file in Europa_list:
    R_moon = 1560.8 #km

if flyby_file in Callisto_list:
    R_moon = 2410.3 #km

if flyby_file in Ganymede_list:
    R_moon = 2631 #km

if flyby_file in Io_list:
    R_moon = 1821.6 #km

moon_full = moon_full_names.get(moon_name.upper(), moon_name.capitalize())

# Final formatted title string
title_label = f"Orbit {orbit_number} of {moon_full}"

df = pd.read_csv(flyby_file, sep='\s+', header=None)

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

# Convert the time to datetime 
timestamps = pd.to_datetime(time)

# Convert timestamps to UTC hours (relative to the start time)
start_time = timestamps[0]
time_diff = timestamps - start_time  # This will give a Timedelta
timeUTC = mdates.date2num(df[0])

flyby_date = df[0].iloc[0].date()
flyby_date_str = flyby_date.strftime("%d %B %Y")

colat = np.radians(90-theta)
east_long = np.radians(phi)


jm.Internal.Config(Model="jrm33",CartesianIn=False,CartesianOut=False)
Bjr,Bjt,Bjp = jm.Internal.Field(r, colat, east_long)

#be sure to configure external field model prior to tracing
jm.Con2020.Config(equation_type='analytic')
#this may also become necessary with internal models in future, e.g.
#setting the model degree

sph_model = con2020.Model(equation_type='analytic', CartesianIn=False,CartesianOut=False)


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

X_min = 10   # half-window size in minutes (adjust as needed)
Y = 2        # polynomial degree (adjust as needed)

pf = pd.read_csv(background_file, sep='\s+', header=None)


x = pf[5].to_numpy()  # 3rd last column for x
y = pf[6].to_numpy()  # 2nd last column for y
z = pf[7].to_numpy()  # Last column for z

ind_r = np.sqrt(x**2 + y**2 + z**2)
ind_t = np.arccos(z/r)
ind_p = np.arctan2(x, y)


# Calculate the distance from the surface of the moon
distance_from_surface = ind_r - 1  # Distance from the surface, moon centre of mass 0,0,0)

r = distance_from_surface * R_moon

ind_r *= R_moon

# Convert times to seconds relative to start
time_sec = (timestamps - start_time).total_seconds().to_numpy()

# Closest approach (already computed)
closest_idx = np.argmin(distance_from_surface)
t_ca_sec = time_sec[closest_idx]  # seconds from start

# Exclusion window
delta_t = X_min * 60.0  # seconds
t_start = t_ca_sec - delta_t
t_end   = t_ca_sec + delta_t

# Compute datetime values
t_start_dt = start_time + pd.to_timedelta(t_start, unit='s')
t_end_dt   = start_time + pd.to_timedelta(t_end, unit='s')

# Convert to matplotlib date numbers
t_data_start = mdates.date2num(timestamps[0])
t_data_end   = mdates.date2num(timestamps[-1])
t_start_num  = mdates.date2num(t_start_dt)
t_end_num    = mdates.date2num(t_end_dt)

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


conductivity_sets = [ #  (conductivity [S/m], g11, h11)
    (1.0, -6.2, 90),
    (3.0, -5.5, 95),
    (10, -5.0, 97)
]

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig.suptitle(f'Galileo MAG for {title_label} on {flyby_date_str}', fontsize = 15)

# Original data
axs[0].plot(timeUTC, Bx, 'k', label='Bx data')
axs[1].plot(timeUTC, By, 'k', label='By data')
axs[2].plot(timeUTC, Bz, 'k', label='Bz data')

# Use a color palette for different conductivities
colors = plt.cm.viridis(np.linspace(0, 1, len(conductivity_sets)))

for i, (sigma, g11, h11) in enumerate(conductivity_sets):
    # Compute induced fields
    B_r_ind = 2 * (R_moon / ind_r) ** 3 * np.sin(ind_t) * (g11 * np.cos(ind_p) + h11 * np.sin(ind_p))
    B_t_ind = -1 * (R_moon / ind_r) ** 3 * np.cos(ind_t) * (g11 * np.cos(ind_p) + h11 * np.sin(ind_p))
    B_p_ind = (R_moon / ind_r) ** 3 * (g11 * np.sin(ind_p) - h11 * np.cos(ind_p))

    # Add background fit (as before)
    B_r_ind = B_r_ind + By_fit
    B_t_ind = B_t_ind + Bz_fit
    B_p_ind = B_p_ind + Bx_fit

    # Plot induced fields with conductivity in legend
    label = f"$\sigma$ = {sigma} S/m"
    axs[0].plot(timeUTC, B_p_ind, '--', color=colors[i], label=label)
    axs[1].plot(timeUTC, B_r_ind, '--', color=colors[i], label=label)
    axs[2].plot(timeUTC, B_t_ind, '--', color=colors[i], label=label)

axs[0].set_ylabel('Bx (nT)', fontsize = 15)
axs[1].set_ylabel('By (nT)', fontsize = 15)
axs[2].set_ylabel('Bz (nT)', fontsize = 15)
axs[2].set_xlabel('Time (UTC)', fontsize = 15)

for ax in axs:
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.tick_params(axis='x', rotation=45)
    ax.axvspan(t_data_start, t_start_num, color='gray', alpha=0.3)
    ax.axvspan(t_end_num, t_data_end, color='gray', alpha=0.3)

plt.tight_layout()
plt.show()
