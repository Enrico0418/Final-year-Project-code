# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:30:55 2025

@author: Enrico
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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

df = pd.read_csv(flyby_file, delim_whitespace=True, header=None)

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
df[0] = pd.to_datetime(df[0], errors='coerce')
df['time'] = df[0].dt.strftime('%H:%M')
time = df[0].to_numpy()  # First column for time
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
timeUTC = time_diff.total_seconds() / 3600  # Convert to hours

flyby_duration_hours = (timeUTC[-1] - timeUTC[0])


#timeUTC = df['time']

# Set up the figure with multiple subplots (5 rows, 1 column for magnetic field plots and distance)
fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
fig.suptitle(f'Magnetic field measurments of Galileo during {title_label}', y=0.99)

# Plot for Bx
axs[0].plot(timeUTC, BX, label='Bx (nT)', color='black')
axs[0].set_ylabel('Bx (nT)')
axs[0].tick_params(axis='x', rotation=45)
axs[0].grid(True)

# Plot for By
axs[1].plot(timeUTC, BY, label='By (nT)', color='blue')
axs[1].set_ylabel('By (nT)')
axs[1].tick_params(axis='x', rotation=45)
axs[1].grid(True)  # Add grid to the second subplot

# Plot for Bz
axs[2].plot(timeUTC, BZ, label='Bz (nT)', color='purple')
axs[2].set_ylabel('Bz (nT)')
axs[2].tick_params(axis='x', rotation=45)
axs[2].grid(True)  # Add grid to the third subplot

# Plot for B Total
axs[3].plot(timeUTC, BTot, label='B Total (nT)', color='red', linestyle='--')
axs[3].set_ylabel('B Total (nT)')
axs[3].tick_params(axis='x', rotation=45)
axs[3].grid(True)  # Add grid to the fourth subplot

# Plot for Distance from Europa's surface
axs[4].plot(timeUTC, distance_from_surface, label='Distance from Europa surface (km)', color='blue')
axs[4].set_xlabel('Time (UTC Hours)')
axs[4].set_ylabel('Distance ($R_E$)')
axs[4].tick_params(axis='x', rotation=45)
axs[4].grid(True)  # Add grid to the fifth subplot

# Add a vertical dashed line for Closest Approach (CA) on each subplot
closest_approach_time = timeUTC[np.argmin(distance_from_surface)]  # Find time of closest approach
for i, ax in enumerate(axs):
    ax.axvline(x=closest_approach_time, color='black', linestyle='--', label='Closest Approach (CA)' if i == 0 else None)

# Adjust layout for magnetic field and distance plots (remove extra space)
axs[-1].set_xlim(0,flyby_duration_hours)
plt.tight_layout()

# Now, plot the 3D trajectory projections in a separate figure
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(f'Trajectory of Galileo Flyby {title_label}', y=0.97)

# Plot the trajectory in the xy-plane (no gradient, simple scatter plot)
axs2[0].scatter(x, y, color='black', s=10, label='Trajectory (xy-plane)')
axs2[0].set_xlabel('X ($R_E$)')
axs2[0].set_ylabel('Y ($R_E$)')
moon_circle = Circle((0, 0), 1, color='black', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[0].add_patch(moon_circle)  # Add the moon
axs2[0].set_xlim([-6.5, 6.5])  # Set the range of x-axis from -6 to 6
axs2[0].set_ylim([-6.5, 6.5])  # Set the range of y-axis from -6 to 6
axs2[0].grid(True)  # Add grid to the first 3D projection plot

# Plot the trajectory in the yz-plane (no gradient, simple scatter plot)
axs2[1].scatter(y, z, color='black', s=10, label='Trajectory (yz-plane)')
axs2[1].set_xlabel('Y ($R_E$)')
axs2[1].set_ylabel('Z ($R_E$)')
moon_circle = Circle((0, 0), 1, color='black', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[1].add_patch(moon_circle)  # Add the moon
axs2[1].set_xlim([-4, 4])  # Set the range of y-axis from -6 to 6
axs2[1].set_ylim([-4, 4])  # Set the range of z-axis from -6 to 6
axs2[1].grid(True)  # Add grid to the second 3D projection plot

# Plot the trajectory in the xz-plane (no gradient, simple scatter plot)
axs2[2].scatter(x, z, color='black', s=10, label='Trajectory (xz-plane)')
axs2[2].set_xlabel('X ($R_E$)')
axs2[2].set_ylabel('Z ($R_E$)')
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
X_min = 5   # half-window size in minutes (adjust as needed)
Y = 1       # polynomial degree (adjust as needed)

# --- TIME HANDLING ---
# Convert times to seconds relative to start
time_sec = (timestamps - start_time).total_seconds().to_numpy()

# Closest approach (already computed)
t_ca_sec = time_sec[np.argmin(distance_from_surface)]

# Exclusion window
delta_t = X_min * 60.0  # seconds
mask = (time_sec < t_ca_sec - delta_t) | (time_sec > t_ca_sec + delta_t)

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

# --- SPATIAL LENGTH OF EXCLUDED INTERVAL ---
  # km/s (approx constant near CA)
speed = 8 # in km/s


# Radial distance from Europa's center (km) and altitude (km)
r = np.sqrt(x**2 + y**2 + z**2) * R_moon
altitude = r - R_moon

# Exclusion window times
t_start = t_ca_sec - delta_t
t_end   = t_ca_sec + delta_t

# Interpolate altitude and coordinates at exclusion start
alt_start = np.interp(t_start, time_sec, altitude)
x_start   = np.interp(t_start, time_sec, x)
y_start   = np.interp(t_start, time_sec, y)
z_start   = np.interp(t_start, time_sec, z)

# Interpolate altitude and coordinates at exclusion end
alt_end = np.interp(t_end, time_sec, altitude)
x_end   = np.interp(t_end, time_sec, x)
y_end   = np.interp(t_end, time_sec, y)
z_end   = np.interp(t_end, time_sec, z)

# Print results
print("=== Exclusion Window Boundaries ===")
print(f"Start: Altitude = {alt_start:.1f} km ({alt_start/R_moon:.2f} R units), "
      f"Coords = ({x_start:.3f}, {y_start:.3f}, {z_start:.3f}) R units")
print(f"End:   Altitude = {alt_end:.1f} km ({alt_end/R_moon:.2f} R units), "
      f"Coords = ({x_end:.3f}, {y_end:.3f}, {z_end:.3f}) R units")



fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axs[0].plot(time_sec/3600, BX, 'k', label='Bx data')
axs[0].plot(time_sec/3600, Bx_fit, 'r--', label='Bx fit')
axs[0].legend(); axs[0].set_ylabel('Bx (nT)')

axs[1].plot(time_sec/3600, BY, 'b', label='By data')
axs[1].plot(time_sec/3600, By_fit, 'r--', label='By fit')
axs[1].legend(); axs[1].set_ylabel('By (nT)')

axs[2].plot(time_sec/3600, BZ, 'm', label='Bz data')
axs[2].plot(time_sec/3600, Bz_fit, 'r--', label='Bz fit')
axs[2].legend(); axs[2].set_ylabel('Bz (nT)')
axs[2].set_xlabel('Time since start (hours)')

axs[3].plot(time_sec/3600, BTot, 'r', label='B data')
axs[3].plot(time_sec/3600, B_fit, 'r--', label='B fit')
axs[3].legend(); axs[3].set_ylabel('B (nT)')
axs[3].set_xlabel('Time since start (hours)')

axs[-1].set_xlim(0,flyby_duration_hours)
plt.tight_layout()
plt.show()

""" Residual calculations """

Bx_res = BX-Bx_fit
By_res = BY-By_fit
Bz_res = BZ-Bz_fit
B_res = BTot-B_fit

# --- PLOT DIAGNOSTIC ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axs[0].plot(time_sec/3600, Bx_res, 'k', label='Bx data')
axs[0].legend(); axs[0].set_ylabel('Bx (nT)')

axs[1].plot(time_sec/3600, By_res, 'b', label='By data')
axs[1].legend(); axs[1].set_ylabel('By (nT)')

axs[2].plot(time_sec/3600, Bz_res, 'm', label='Bz data')
axs[2].legend(); axs[2].set_ylabel('Bz (nT)')

axs[3].plot(time_sec/3600, B_res, 'r', label='B data')
axs[3].legend(); axs[3].set_ylabel('B (nT)')

axs[-1].set_xlim(0,flyby_duration_hours)

plt.tight_layout()
plt.show()

output_filename = f"Residuals_{title_label.replace(' ', '_')}.txt"

# Prepare a dataframe to hold residuals
residuals_data = {  # Convert seconds to hours
    "Bx_residuals (nT)": Bx_res,
    "By_residuals (nT)": By_res,
    "Bz_residuals (nT)": Bz_res,
    "B_residuals (nT)": B_res
}

# Create a DataFrame from the residuals
residuals_df = pd.DataFrame(residuals_data)

# Save the DataFrame to a .txt file
residuals_df.to_csv(output_filename, sep='\t', index=False)

print(f"Residuals saved to {output_filename}")
