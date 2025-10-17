# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:30:55 2025

@author: Enrico
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from numpy.polynomial import Polynomial

flyby_file = input("Enter flyby file").strip()

Europa_list = {"ORB04_EUR_EPHIO.TAB", "ORB11_EUR_EPHIO.TAB", 
               "ORB12_EUR_EPHIO.TAB", "ORB14_EUR_EPHIO.TAB", 
               "ORB15_EUR_EPHIO.TAB", "ORB17_EUR_EPHIO.TAB", 
               "ORB19_EUR_EPHIO.TAB", "ORB25_EUR_EPHIO.TAB", 
               "ORB25_EUR_EPHIO.TAB",}
Callisto_list = {"ORB03_CALL_CPHIO.TAB", "ORB09_CALL_CPHIO.TAB", 
                 "ORB10_CALL_CPHIO.TAB", "ORB21_CALL_CPHIO.TAB", 
                 "ORB22_CALL_CPHIO.TAB", "ORB23_CALL_CPHIO.TAB", 
                 "ORB30_CALL_CPHIO.TAB"}
Ganymede_list = {"ORB01_GAN_GPHIO.TAB", "ORB02_GAN_GPHIO.TAB", 
                 "ORB07_GAN_GPHIO.TAB", "ORB08_GAN_GPHIO.TAB", 
                 "ORB09_GAN_GPHIO.TAB", "ORB12_GAN_GPHIO.TAB", 
                 "ORB28_GAN_GPHIO.TAB", "ORB29_GAN_GPHIO.TAB"}
Io_list = {"ORB00_IO_IPHIO.TAB", "ORB24_IO_IPHIO.TAB", 
           "ORB27_IO_IPHIO.TAB", "ORB31_IO_IPHIO.TAB", 
           "ORB32_IO_IPHIO.TAB"}

df = pd.read_csv(flyby_file, sep='\s+', header=None)

R_moon = 0

if flyby_file in Europa_list:
    R_moon = 1560.8 #km

if flyby_file in Callisto_list:
    R_moon = 2410.3 #km

if flyby_file in Ganymede_list:
    R_moon = 2631 #km

if flyby_file in Io_list:
    R_moon = 1821.6 #km

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

# Assign all the columns to a variable
df[0] = df[0].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
df['time'] = df[0].dt.strftime('%H:%M')  # Optional human-readable for table
time = df[0].to_numpy()
BX = df[1].to_numpy()   # Second column for Bx
BY = df[2].to_numpy()   # Third column for By
BZ = df[3].to_numpy()   # Fourth column for Bz
BTot = df[4].to_numpy() # Fifth column for B Total
x = df[5].to_numpy()  # 3rd last column for x
y = df[6].to_numpy()  # 2nd last column for y
z = df[7].to_numpy()  # Last column for z

# Calculate the distance from the surface of Europa
distance_from_surface = np.sqrt(x**2 + y**2 + z**2) - 1  # Distance from the surface, Europa centre of mass 0,0,0)

# Convert the time to datetime (assuming it's already in a suitable format)
timestamps = pd.to_datetime(time)

# Convert timestamps to UTC hours (relative to the start time)
start_time = timestamps[0]
time_diff = timestamps - start_time  # This will give a Timedelta
timeUTC = mdates.date2num(df[0])


flyby_date = df[0].iloc[0].date()
flyby_date_str = flyby_date.strftime("%d %B %Y")

#timeUTC = df['time']

# Set up the figure with multiple subplots (5 rows, 1 column for magnetic field plots and distance)
fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
fig.suptitle(f'Galileo MAG for {title_label} on {flyby_date_str}')
plt.tight_layout(pad=2.0)  # reduce padding between title and subplots

# Plot for Bx
axs[0].plot(timeUTC, BX, 'k', label='Bx (nT)')
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs[0].xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,5)))

axs[0].plot(timeUTC, BX, label='Bx (nT)', color='black')
axs[0].set_ylabel('Bx (nT)')
axs[0].tick_params(axis='x', rotation=45)
axs[0].grid(True)

# Plot for By
axs[1].plot(timeUTC, BY, label='By (nT)', color='black')
axs[1].set_ylabel('By (nT)')
axs[1].tick_params(axis='x', rotation=45)
axs[1].grid(True)  # Add grid to the second subplot

# Plot for Bz
axs[2].plot(timeUTC, BZ, label='Bz (nT)', color='black')
axs[2].set_ylabel('Bz (nT)')
axs[2].tick_params(axis='x', rotation=45)
axs[2].grid(True)  # Add grid to the third subplot

# Plot for B Total
axs[3].plot(timeUTC, BTot, label='B Total (nT)', color='black', linestyle='--')
axs[3].set_ylabel('B Total (nT)')
axs[3].tick_params(axis='x', rotation=45)
axs[3].grid(True)  # Add grid to the fourth subplot

# Plot for Distance from Europa's surface
axs[4].plot(timeUTC, distance_from_surface, label='Distance from Europa surface (km)', color='blue')
axs[4].set_xlabel('Time (UTC Hours)')
axs[4].set_ylabel('Distance From Surface($R_Moon$)')
axs[4].tick_params(axis='x', rotation=45)
axs[4].grid(True)  # Add grid to the fifth subplot

# Add a vertical dashed line for Closest Approach (CA) on each subplot
closest_approach_time = timeUTC[np.argmin(distance_from_surface)]  # Find time of closest approach
for i, ax in enumerate(axs):
    ax.axvline(x=closest_approach_time, color='black', linestyle='--', label='Closest Approach (CA)' if i == 0 else None)



# Now, plot the 3D trajectory projections in a separate figure
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(f'Galileo MAG for {title_label} on {flyby_date_str}')

# Plot the trajectory in the xy-plane (no gradient, simple scatter plot)
axs2[0].scatter(x, y, color='black', s=10, label='Trajectory (xy-plane)')
axs2[0].set_xlabel('X ($R_Moon$)')
axs2[0].set_ylabel('Y ($R_Moon$)')
moon_circle = Circle((0, 0), 1, color='black', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[0].add_patch(moon_circle)  # Add the moon
axs2[0].set_xlim([-6.5, 6.5])  # Set the range of x-axis from -6 to 6
axs2[0].set_ylim([-6.5, 6.5])  # Set the range of y-axis from -6 to 6
axs2[0].grid(True)  # Add grid to the first 3D projection plot

# Plot the trajectory in the yz-plane (no gradient, simple scatter plot)
axs2[1].scatter(y, z, color='black', s=10, label='Trajectory (yz-plane)')
axs2[1].set_xlabel('Y ($R_Moon$)')
axs2[1].set_ylabel('Z ($R_Moon$)')
moon_circle = Circle((0, 0), 1, color='black', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[1].add_patch(moon_circle)  # Add the moon
axs2[1].set_xlim([-4, 4])  # Set the range of y-axis from -6 to 6
axs2[1].set_ylim([-4, 4])  # Set the range of z-axis from -6 to 6
axs2[1].grid(True)  # Add grid to the second 3D projection plot

# Plot the trajectory in the xz-plane (no gradient, simple scatter plot)
axs2[2].scatter(x, z, color='black', s=10, label='Trajectory (xz-plane)')
axs2[2].set_xlabel('X ($R_Moon$)')
axs2[2].set_ylabel('Z ($R_Moon$)')
moon_circle = Circle((0, 0), 1, color='black', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[2].add_patch(moon_circle)  # Add the moon
axs2[2].set_xlim([-6.5, 6.5])  # Set the range of x-axis from -6 to 6
axs2[2].set_ylim([-6.5, 6.5])  # Set the range of z-axis from -6 to 6
axs2[2].grid(True)  # Add grid to the third 3D projection plot

# Adjust layout for the 3D projection plots
plt.tight_layout()

# Show both figures
plt.show()

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
coeffs_Bx = Polynomial.fit(time_sec[mask], BX[mask], deg=Y).convert().coef
coeffs_By = Polynomial.fit(time_sec[mask], BY[mask], deg=Y).convert().coef
coeffs_Bz = Polynomial.fit(time_sec[mask], BZ[mask], deg=Y).convert().coef

# Evaluate fitted background over full time axis
Bx_fit = np.polyval(coeffs_Bx[::-1], time_sec)  # reverse for np.polyval
By_fit = np.polyval(coeffs_By[::-1], time_sec)
Bz_fit = np.polyval(coeffs_Bz[::-1], time_sec)
B_fit = np.sqrt(Bx_fit**2 + By_fit**2 + Bz_fit**2)

# --- RADIAL DISTANCE AND ALTITUDE ---
# km (approx constant near CA)
speed = 8  # km/s

r = np.sqrt(x**2 + y**2 + z**2) * R_moon  # radial distance from moon center (km)
altitude = r - R_moon                      # altitude above moon surface

# Interpolate altitude and coordinates at exclusion start, CA, and end
alt_start = np.interp(t_start, time_sec, altitude)
x_start   = np.interp(t_start, time_sec, x)
y_start   = np.interp(t_start, time_sec, y)
z_start   = np.interp(t_start, time_sec, z)

alt_ca = np.interp(t_ca_sec, time_sec, altitude)
x_ca = np.interp(t_ca_sec, time_sec, x)
y_ca = np.interp(t_ca_sec, time_sec, y)
z_ca = np.interp(t_ca_sec, time_sec, z)

alt_end = np.interp(t_end, time_sec, altitude)
x_end   = np.interp(t_end, time_sec, x)
y_end   = np.interp(t_end, time_sec, y)
z_end   = np.interp(t_end, time_sec, z)

# Print results
print("=== Exclusion Window Boundaries ===")
print(f"Start: Altitude = {alt_start:.1f} km ({alt_start/R_moon:.2f} R units), "
      f"Coords = ({x_start:.3f}, {y_start:.3f}, {z_start:.3f}) R units")
print(f"Closest Approach: Altitude= {alt_ca:.1f} km ({alt_ca/R_moon:.2f} R units), "
      f"Coords= ({x_ca:.3f}, {y_ca:.3f}, {z_ca:.3f}) R units")
print(f"End:   Altitude = {alt_end:.1f} km ({alt_end/R_moon:.2f} R units), "
      f"Coords = ({x_end:.3f}, {y_end:.3f}, {z_end:.3f}) R units")

# --- PLOTS OF FIT AND DATA (with datetime axis) ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.suptitle(f'Galileo MAG for {title_label} on {flyby_date_str}')

# Bx
axs[0].plot(timeUTC, BX, 'k', label='Bx data')
axs[0].plot(timeUTC, Bx_fit, 'r--', label='Bx fit')
axs[0].set_ylabel('Bx (nT)')
axs[0].legend()
axs[0].grid(True)

# By
axs[1].plot(timeUTC, BY, 'k', label='By data')
axs[1].plot(timeUTC, By_fit, 'r--', label='By fit')
axs[1].set_ylabel('By (nT)')
axs[1].legend()
axs[1].grid(True)

# Bz
axs[2].plot(timeUTC, BZ, 'k', label='Bz data')
axs[2].plot(timeUTC, Bz_fit, 'r--', label='Bz fit')
axs[2].set_ylabel('Bz (nT)')
axs[2].legend()
axs[2].grid(True)

# Total B
axs[3].plot(timeUTC, BTot, 'k', label='B data')
axs[3].plot(timeUTC, B_fit, 'r--', label='B fit')
axs[3].set_ylabel('B (nT)')
axs[3].legend()
axs[3].grid(True)
axs[3].set_xlabel('Time (UTC)')


# Format x-axis with hours and minutes
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.tick_params(axis='x', rotation=45)

# Add vertical line for closest approach
closest_approach_time = timeUTC[closest_idx]
for i, ax in enumerate(axs):
    ax.axvline(x=closest_approach_time, color='black', linestyle='--',
               label='Closest Approach (CA)' if i == 0 else None)

plt.tight_layout()
plt.show()

# --- RESIDUAL CALCULATION ---
Bx_res = BX - Bx_fit
By_res = BY - By_fit
Bz_res = BZ - Bz_fit
B_res = BTot - B_fit

# --- RESIDUAL PLOTS (with datetime axis) ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.suptitle(f'B-field residuals for {title_label} on {flyby_date_str}')

# Bx residual
axs[0].plot(timeUTC, Bx_res, 'k', label='Bx residual')
axs[0].set_ylabel('Bx (nT)')
axs[0].legend()
axs[0].grid(True)

# By residual
axs[1].plot(timeUTC, By_res, 'k', label='By residual')
axs[1].set_ylabel('By (nT)')
axs[1].legend()
axs[1].grid(True)

# Bz residual
axs[2].plot(timeUTC, Bz_res, 'k', label='Bz residual')
axs[2].set_ylabel('Bz (nT)')
axs[2].legend()
axs[2].grid(True)

# Total B residual
axs[3].plot(timeUTC, B_res, 'k', label='B residual')
axs[3].set_ylabel('B (nT)')
axs[3].legend()
axs[3].grid(True)
axs[3].set_xlabel('Time (UTC)')

# Format x-axis with hours and minutes
for ax in axs:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))
    ax.tick_params(axis='x', rotation=45)

# Vertical line for closest approach
for i, ax in enumerate(axs):
    ax.axvline(x=closest_approach_time, color='black', linestyle='--',
               label='Closest Approach (CA)' if i == 0 else None)

plt.tight_layout()
plt.show()


# --- SAVE RESIDUALS TO FILE ---
output_filename = f"Residuals_{title_label.replace(' ', '_')}.txt"
residuals_data = {
    "Bx_residuals (nT)": Bx_res,
    "By_residuals (nT)": By_res,
    "Bz_residuals (nT)": Bz_res,
    "B_residuals (nT)": B_res
}
residuals_df = pd.DataFrame(residuals_data)
residuals_df.to_csv(output_filename, sep='\t', index=False)
print(f"Residuals saved to {output_filename}")
