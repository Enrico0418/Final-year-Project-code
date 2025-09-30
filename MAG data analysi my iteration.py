# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:30:55 2025

@author: Enrico
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

# Load the data from the .TAB file
df = pd.read_csv('ORB14_EUR_EPHIO.TAB', delim_whitespace=True, header=None)

# Assign all the columns to a variable
time = df[0].to_numpy()  # First column for time
BX = df[1].to_numpy()   # Second column for Bx
BY = df[2].to_numpy()   # Third column for By
BZ = df[3].to_numpy()   # Fourth column for Bz
BTot = df[4].to_numpy() # Fifth column for B Total
x = df[5].to_numpy()  # 3rd last column for x
y = df[6].to_numpy()  # 2nd last column for y
z = df[7].to_numpy()  # Last column for z

# Calculate the distance from the surface of Europa
distance_from_surface = np.sqrt(x**2 + y**2 + z**2) - 1  # Subtract Europa's radius (1)

# Convert the time to datetime (assuming it's already in a suitable format)
timestamps = pd.to_datetime(time)

# Convert timestamps to UTC hours (relative to the start time)
start_time = timestamps[0]
time_diff = timestamps - start_time  # This will give a Timedelta
timeUTC = time_diff.total_seconds() / 3600  # Convert to hours

# Set up the figure with multiple subplots (5 rows, 1 column for magnetic field plots and distance)
fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)
fig.suptitle('Magnetic field measurments of Galileo MAG during ', y=0.99)

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
plt.tight_layout()

# Now, plot the 3D trajectory projections in a separate figure
fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Trajectory of Galileo Flyby', y=0.97)

# Plot the trajectory in the xy-plane (no gradient, simple scatter plot)
axs2[0].scatter(x, y, color='blue', s=10, label='Trajectory (xy-plane)')
axs2[0].set_xlabel('X ($R_E$)')
axs2[0].set_ylabel('Y ($R_E$)')
moon_circle = Circle((0, 0), 1, color='orange', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[0].add_patch(moon_circle)  # Add the moon
axs2[0].set_xlim([-6.5, 6.5])  # Set the range of x-axis from -6 to 6
axs2[0].set_ylim([-6.5, 6.5])  # Set the range of y-axis from -6 to 6
axs2[0].grid(True)  # Add grid to the first 3D projection plot

# Plot the trajectory in the yz-plane (no gradient, simple scatter plot)
axs2[1].scatter(y, z, color='red', s=10, label='Trajectory (yz-plane)')
axs2[1].set_xlabel('Y ($R_E$)')
axs2[1].set_ylabel('Z ($R_E$)')
moon_circle = Circle((0, 0), 1, color='orange', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[1].add_patch(moon_circle)  # Add the moon
axs2[1].set_xlim([-4, 4])  # Set the range of y-axis from -6 to 6
axs2[1].set_ylim([-4, 4])  # Set the range of z-axis from -6 to 6
axs2[1].grid(True)  # Add grid to the second 3D projection plot

# Plot the trajectory in the xz-plane (no gradient, simple scatter plot)
axs2[2].scatter(x, z, color='green', s=10, label='Trajectory (xz-plane)')
axs2[2].set_xlabel('X ($R_E$)')
axs2[2].set_ylabel('Z ($R_E$)')
moon_circle = Circle((0, 0), 1, color='orange', fill=False, linewidth=2, label='Europa')  # Create a new moon circle for this plot
axs2[2].add_patch(moon_circle)  # Add the moon
axs2[2].set_xlim([-6.5, 6.5])  # Set the range of x-axis from -6 to 6
axs2[2].set_ylim([-6.5, 6.5])  # Set the range of z-axis from -6 to 6
axs2[2].grid(True)  # Add grid to the third 3D projection plot

# Adjust layout for the 3D projection plots
plt.tight_layout()

# Show both figures
plt.show()
