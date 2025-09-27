# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 15:02:42 2025

@author: Enrico
"""

from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

def format_data(orbit_path, moon_name):    
    """
    Format .TAB orbit data into a pandas dataframe.

    Parameters
    ----------

    orbit_path: str
        File path where .TAB orbit file is stored.

    moon_name: "Ganymede", "Europa", "Io", "Callisto"
        Name of the moon over where the orbit occured.

    Returns
    -------

    orbit_data: pd.Dataframe
        Formatted Dataframe containing Time, B_x, B_y, B_z, B, X, Y, Z, Distance, Moon and start date columns.
    """
    if moon_name not in ["Ganymede", "Europa", "Io", "Callisto"]:
        raise ValueError("The only valid moon names are Ganymede, Europa, Io and Callisto")
    # Header units: absolute time, magnetic fields in nT, coordinates of Moon radii
    header = ['Time', 'B_x', 'B_y', 'B_z', 'B', 'X', 'Y', 'Z']

    # Create pandas dataframe
    orbit_data = pd.read_csv(orbit_path, sep='\s+', names=header)

    # Add column distance to Moon's Surface in Moon radii
    orbit_data["Distance"] = np.sqrt(orbit_data["X"]**2 + orbit_data["Y"]**2 + orbit_data["Z"]**2) - 1
    
    orbit_data["moon"] = moon_name
    orbit_data["start date"] = datetime.strptime(orbit_data['Time'][0], '%Y-%m-%dT%H:%M:%S.%f').strftime('%B %e, %Y')
    # Convert Time strings into np.datetime64 dates 
    orbit_data["Time"] = orbit_data["Time"].astype("datetime64[ms]")

    return orbit_data

def wake_dates(orbit_data):
    """
    Find times at which spacecraft enters and leaves the solar wind wake of the moon.

    Parameters
    ----------

    orbit_data: pd.Dataframe
        Formatted .TAB orbit data from format_data func.

    Returns
    -------

    start, stop: np.datetime64 or np.nan
        Times at which the spacecraft enters/leaves the wake of the moon.
    """
    # Spacecraft in wake if X is positive and square root of sum of Y squared and Z squared is below 1
    wake_cond = (orbit_data['X'] > 0) & (np.sqrt(orbit_data['Y']**2 + orbit_data['Z']**2) < 1)

    # Find start and stop times
    start, stop = orbit_data[wake_cond]["Time"].min(), orbit_data[wake_cond]["Time"].max()

    # Check that a value has been found, otherwise return nan
    if pd.isnull([start, stop]).any():
        start, stop = np.nan, np.nan
    
    return start, stop

def choose_rad_symbol(orbit_data):
    """
    Select Moon radius symbol to be shown on plots.

    Parameters
    ----------

    orbit_data: pd.Dataframe
        Formatted .TAB data from format_data func.

    Returns
    -------

    rad: str
        Moon symbol.
    """
    if orbit_data['moon'][0] == 'Ganymede':
        rad = 'GAN'

    if orbit_data['moon'][0] == 'Europa':
        rad = 'E'

    if orbit_data['moon'][0] == 'Callisto':
        rad = 'CAL'

    if orbit_data['moon'][0] == 'Io':
        rad = 'Io'

    return rad

def plot_mag(orbit_data, fig, gs, i, mag, color='blue'):
    """
    Plot Magnetic field time series on a 5-tile GridSpec Figure.

    Parameters
    ----------

    orbit_data: pd.DataFrame
        Formatted .TAB orbit data from format_data func.

    fig: matplotlib.Figure
        Figure upon which to draw the axes.

    gs: matplotlib.GridSpec(5, 1)
        GridSpec grid on which to plot the axes.
    
    i: int
        Index of the GridSpec grid on which to plot the axes.
    
    mag: "B_x", "B_y", "B_z" or "B"
        Magnetic field column of orbit_data representing flux to be plotted.
    
    color: str
        Color of magnetic flux on the plot.

    Returns
    -------

    ax: matplotlib.Axes
        Annotated time series plot of magnetic flux. 
    """

    ax = fig.add_subplot(gs[i])

    # Plot mag time series
    ax.plot(orbit_data["Time"], orbit_data[f'{mag}'], color)

    # Find and plot closest approach
    ca = orbit_data[orbit_data["Distance"] == orbit_data["Distance"].min()]["Time"]
    ax.vlines(ca, 0, 1, 'black', transform=ax.get_xaxis_transform(), lw=1, ls='--')

    # Plot wake area
    geo_wake_area = ax.axvspan(*wake_dates(orbit_data), color='grey', alpha=0.5)

    # Add CA and wake area annotations if plotting B_x
    if mag == 'B_x':
        ax.annotate('Geometric wake', (.5, 1), xycoords=geo_wake_area, ha='center', va='bottom', color='grey')
        ax.annotate('CA', (ca, 1), xycoords=('data', 'axes fraction'), ha='center', va='bottom', color='black')

    ax.set_ylabel(f'${mag}$ [nT]')

    # Plot every 6th minute of time series, do not show any ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter(''))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,3), interval=2))

    # Ticks, margins, grid formatting
    ax.tick_params(direction='in')
    ax.margins(x=0, y=0)
    ax.grid(color='grey', alpha=0.3, ls='--')

    return ax

def plot_distance(orbit_data, fig, gs, i, text_offset=-35):
    """
    Plot Spacecraft Distance to Moon surface on a 5-tile GridSpec Figure.

    Parameters
    ----------

    orbit_data: pd.DataFrame
        Formatted .TAB orbit data from format_data func.

    fig: matplotlib.Figure
        Figure upon which to draw the axes.

    gs: matplotlib.GridSpec(5, 1)
        GridSpec grid on which to plot the axes.
    
    i: int
        Index of the GridSpec grid on which to plot the axes.
    
    mag: "B_x", "B_y", "B_z" or "B"
        Magnetic field column of orbit_data representing flux to be plotted.
    
    text_offset: int
        Y-offset in pixels from bottom of plot to print XYZ coordinates under timestamps.

    Returns
    -------

    ax: matplotlib.Axes
        Annotated time series plot of spacecraft distance to Moon's surface. Also show timestamps and coordinates XYZ coordinates at each time stamp.
    """
    # Choose rad symbol
    rad = choose_rad_symbol(orbit_data)

    ax = fig.add_subplot(gs[i])

    # Plot Spacecraft distance time series
    ax.plot(orbit_data['Time'], orbit_data['Distance'])

    # Find and plot closest approach
    ca = orbit_data["Distance"].min()
    ca_time = orbit_data[orbit_data["Distance"] == ca]["Time"]
    ax.vlines(ca_time, 0, 1, 'black', transform=ax.get_xaxis_transform(), lw=1, ls='--')

    # Plot wake area
    ax.axvspan(*wake_dates(orbit_data), color='grey', alpha=0.5)

    # Show x and y labels
    ax.set_xlabel('UTC')
    ax.set_ylabel(f"Distance [$R_{{{rad}}}$]")

    # Plot every 6th minute of time series, only show hour and minute on ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0,60,3), interval=2))

    # Ticks, margins, grid formatting
    ax.tick_params(direction='in')
    ax.margins(x=0, y=0)
    ax.grid(color='grey', alpha=0.3, ls='--')

    # Line names of Coordinates Table
    ax.annotate(f'$X$[$R_{{{rad}}}$]\n$Y$[$R_{{{rad}}}$]\n$Z$[$R_{{{rad}}}$]', (orbit_data['Time'][0], ca), xytext=(0, text_offset), textcoords='offset points', ha='right', va='top', fontsize=9.1)

    # Annotate X, Y, Z value under each time tick
    for tick in ax.get_xticks():

        # Convert time tick to datetime
        date_tick = np.datetime64(mdates.num2date(tick), 'ms')
        
        # Find nearest time and get row
        nearest_idx = (orbit_data['Time'] - date_tick).abs().idxmin()
        row = orbit_data.loc[nearest_idx]

        # Get X, Y, Z values associated to time tick
        x_val, y_val, z_val = row['X'], row['Y'], row['Z']

        # Format the annotation text
        text = f'{x_val:.2f}\n{y_val:.2f}\n{z_val:.2f}'
         
        # Annotate X, Y, Z value under each time tick
        ax.annotate(text, (tick, ca), xytext=(0, text_offset), textcoords='offset points', ha='center', va='top')

    return ax

def time_labels(orbit_data, points_num):
    """
    Select timestamps and create time labels to be shown on trajectory plots.

    Parameters
    ----------

    orbit_data: pd.DataFrame
        Formatted .TAB orbit data from format_data func.
    
    points_num: int
        Number of timestamps points to be shown on the trajectory plots.

    Returns
    -------
    (x_time, y_time, z_time): tuple of np.array of np.datetime64
        Timestamps points to be shown on plots.

    (x_label, y_label, z_label): tuple of strings
        Selection of every second timestamp from the above arrays.
    
    time_strings: str
        Time label of every second point to be drawn on plots.

    """

    # Find evenly spaced indices
    step = int(len(orbit_data)/points_num)
    start = int(step/2)

    # Select values of x,y,z,time for time labelling
    x_time, y_time, z_time = orbit_data["X"][start::step].to_numpy(), orbit_data["Y"][start::step].to_numpy(), orbit_data["Z"][start::step].to_numpy()

    # Select every second time stamp
    x_label, y_label, z_label = x_time[1::2], y_time[1::2], z_time[1::2]

    # Create label for every second time stamp
    time_sliced = orbit_data["Time"][start::step]
    time_labels = np.asarray(time_sliced[1::2], '<U42')
    time_strings = np.array([
        datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.%f').strftime('%H:%M') 
        for date in time_labels
    ])

    return (x_time, y_time, z_time), (x_label, y_label, z_label), time_strings

def plot_traj(orbit_data, fig, gs, i, coords, flowdir='right', lim=7, points_num=9, date_offsets=[-.8,.2]):
    """
    Plot orbital Mmoon trajectory in XY, YZ or XZ dimensions on one of the 3 tiles of a GridSpec Figure.

    Parameters
    ----------

    orbit_data: pd.DataFrame
        Formatted .TAB orbit data from format_data func.

    fig: matplotlib.Figure
        Figure upon which to draw the axes.

    gs: matplotlib.GridSpec(1, 3)
        GridSpec grid on which to plot the axes.
    
    i: int
        Index of the GridSpec grid on which to plot the axes.

    coords: "XY", "YZ" or "XZ"
        Coordinate system to be plotted.

    flowdir: "right" or "out"
        Direction of the Solar Wind flow.

    lim: int
        X and Y limits of coordinates.
    
    points_num: int
        Number of timestamps points to be shown on the trajectory plots.
    
    date_offsets: list of length 2
        X and Y point offsets of timestamp labels.
    """

    rad = choose_rad_symbol(orbit_data)

    # Retrieve coordinates from the coords string
    coord1, coord2 = coords[0], coords[-1]

    # Flow direction logic to show the correct symbol
    if flowdir == 'right':
        flow_label = '$\longrightarrow$'
    elif flowdir == 'out':
        flow_label = '$\odot$'
    else:
        print('Flow direction keyword invalid')

    # Define x and y limits
    lim1 = -lim
    lim2 = lim

    # Create axis and plot the trajectory data
    ax = fig.add_subplot(gs[i])
    ax.plot(orbit_data[f"{coord1}"], orbit_data[f"{coord2}"])

    # Set axis labels and xy limits
    ax.set_xlabel(f"{coord1} [$R_{{{rad}}}$]")
    ax.set_ylabel(f"{coord2} [$R_{{{rad}}}$]")
    ax.set_xlim(lim1, lim2)
    ax.set_ylim(lim1, lim2)

    # Add Io's trace and shadow on the plot
    if coords == 'XY' or coords == 'XZ':
        ax.add_patch(mpatches.Rectangle((0, -1), lim, 2, facecolor='grey', alpha=0.4, fill=True))
    
    ax.add_patch(mpatches.Circle((0, 0), 1, facecolor='white', edgecolor='black', fill=True))

    # Add centric lines
    ax.hlines(0, lim1, lim2, 'grey', lw=.6, alpha=.6)
    ax.vlines(0, lim1, lim2, 'grey', lw=.6, alpha=.6)

    # Flow legend and symbol
    ax.text((5/7)*lim, (6/7)*lim, 'Flow', fontweight='bold')
    ax.text((4.8/7)*lim, (5/7)*lim, flow_label, fontweight='bold', fontsize='xx-large')

    xyz_coords, xyz_labels, time_strings = time_labels(orbit_data, points_num)

    # Plot trajectory points at specific times
    if coords == 'XY':
        ax.plot(xyz_coords[0], xyz_coords[1], 'k.')
    if coords == 'YZ':
        ax.plot(xyz_coords[1], xyz_coords[2], 'k.')
    if coords == 'XZ':
        ax.plot(xyz_coords[0], xyz_coords[2], 'k.')

    dx, dy = date_offsets

    # Draw time labels
    for label, x_label, y_label, z_label in zip(time_strings, *xyz_labels):
        if coords == 'XY':
            ax.annotate(label, (x_label, y_label), xytext=(x_label + dx, y_label + dy), textcoords='data')
        if coords == 'YZ':
            ax.annotate(label, (y_label, z_label), xytext=(y_label + dx, z_label + dy), textcoords='data')
        if coords == 'XZ':
            ax.annotate(label, (x_label, z_label), xytext=(x_label + dx, z_label + dy), textcoords='data')

    return ax

orbits_io = []
for orb_num in ('00', '24', '27', '31', '32'):

    orb_path = f'ORB{orb_num}_IO_IPHIO.TAB'
    orb_data = format_data(orb_path, 'Io')

    orb_data["number"] = orb_num

    orbits_io.append(orb_data)

orb00_io, orb24_io, orb27_io, orb31_io, orb32_io = orbits_io

orbits_gan = []
for orb_num in ('01', '02', '07', '08', '09', '12', '28', '29'):

    orb_path = f'ORB{orb_num}_GAN_GPHIO.TAB'
    orb_data = format_data(orb_path, 'Ganymede')

    orb_data["number"] = orb_num

    orbits_gan.append(orb_data)

orb00_gan, orb02_gan, orb07_gan, orb08_gan, orb09_gan, orb12_gan, orb28_gan, orb29_gan = orbits_gan

orbits_call = []
for orb_num in ('03', '09', '10', '21', '22', '23', '30'):

    orb_path = f'ORB{orb_num}_CALL_CPHIO.TAB'
    orb_data = format_data(orb_path, 'Callisto')

    orb_data["number"] = orb_num

    orbits_call.append(orb_data)

orb03_call, orb09_call, orb10_call, orb21_call, orb22_call, orb23_call, orb30_call = orbits_call

orbits_eur = []
for orb_num in ('04', '11', '12', '14', '15', '17', '19', '25', '26'):

    orb_path = f'ORB{orb_num}_EUR_EPHIO.TAB'
    orb_data = format_data(orb_path, 'Europa')

    orb_data["number"] = orb_num

    orbits_eur.append(orb_data)

orb04_eur, orb11_eur, orb12_eur, orb14_eur, orb15_eur, orb17_eur, orb19_eur, orb25_eur, orb26_eur = orbits_eur

orbit_data = orb14_eur

fig = plt.figure(figsize=(6,12), dpi=200)
fig.suptitle(f'Magnetic field time series during Galileo Orbit{orbit_data["number"][0]} on {orbit_data["start date"][0]}', ha='center', va='bottom', y=0.9)
gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1], hspace=0)

plot_mag(orbit_data, fig, gs, 0, 'B_x', 'black')
plot_mag(orbit_data, fig, gs, 1, 'B_y', 'blue')
plot_mag(orbit_data, fig, gs, 2, 'B_z', 'purple')
plot_mag(orbit_data, fig, gs, 3, 'B', 'red')
# Distance to surface
ax = plot_distance(orbit_data, fig, gs, 4)

fig = plt.figure(figsize=(15,4.5), dpi=200)
fig.suptitle(f'Trajectory of Galileo during Orbit {orbit_data["number"][0]} on {orbit_data["start date"][0]}', ha='center', va='bottom', y=0.9)
gs = fig.add_gridspec(1, 3, height_ratios=[1], width_ratios=[1, 1, 1], wspace=0.2)

plot_traj(orbit_data, fig, gs, 0, 'XY', flowdir='right', date_offsets=[0,0])
plot_traj(orbit_data, fig, gs, 1, 'YZ', flowdir='out', date_offsets=[-0.5,.2], lim=4)
plot_traj(orbit_data, fig, gs, 2, 'XZ', flowdir='right', date_offsets=[-.8,.2])
