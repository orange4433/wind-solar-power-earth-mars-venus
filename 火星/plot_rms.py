# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:56:10 2024

@author: 19011
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from  matplotlib import patches

# Parameter settings
sol_range = [113, 781]
dt = 0.01
hours = np.arange(0, 24, dt)
size = 20
input_file_east = 'wind_east.npy'
input_file_north = 'wind_north.npy'

# Load data.
RMS_data_east = np.load(input_file_east)
RMS_data_north = np.load(input_file_north)

# Calculate wind speed.
wind_speed = np.sqrt(RMS_data_east**2 + RMS_data_north**2)

sols = np.arange(sol_range[0], sol_range[1])
wind_speed = wind_speed[sol_range[0]:sol_range[1],:]
# Plot settings
fig, ax = plt.subplots(figsize=(8, 8))
cmap = plt.get_cmap('plasma')
vmin = 0  # Set the colorbar minimum to 0.
vmax = 16  # Set the colorbar maximum to 16.

# Create a heatmap.
mesh = ax.pcolormesh(
    hours, sols, wind_speed,
    cmap=cmap,
    norm=colors.Normalize(vmin=vmin, vmax=vmax),
    shading='auto'
)
dayleng = np.array([0,193,372,515,668])
scaleleng = dayleng/dayleng[-1]
lengthscale = np.diff(dayleng)/dayleng[-1]
c = ['green','red','orange','b',]
season = ['Spring','Summer','Fall','Winter',]
for i in range(len(c)):
    p1 = patches.Rectangle((1.,1-scaleleng[i+1]),                # Rectangle lower-left coordinates
                          width = 0.03,                   # Rectangle width
                          height = lengthscale[i],                  # Rectangle height
                          alpha = 0.3,                    # Rectangle opacity
                          facecolor = c[i],            # Whether to fill the rectangle (green)
                          transform=ax.transAxes)    # Keep the rectangle above the image.
    fig.add_artist(p1) 
    plt.text(1.005,1-scaleleng[i+1]+lengthscale[i]/3,season[i],transform=ax.transAxes,rotation=90)
SunRiseSetTime = np.load('./SunRiseSetTime.npy')
sols = SunRiseSetTime[:668,0]
SunRise = SunRiseSetTime[:668,1]
SunSet = SunRiseSetTime[:668,2]
plt.plot(SunRise,sols,color='orange',linestyle='--')
plt.plot(SunSet,sols,color='orange',linestyle='--')
plt.text(0.25,1.01,"Sunrise",transform=ax.transAxes,fontsize=12)
plt.text(0.75,1.01,"Sunset",transform=ax.transAxes,fontsize=12)

# Axis settings
ax.set_xlim(0, 24)
ax.set_ylim(sol_range[0], sol_range[1])
ax.invert_yaxis()
ax.set_xlabel('Local Mean Solar Time (hour)', fontfamily='Times New Roman', fontsize=16)
ax.set_ylabel('Sol (After InSight landing)', fontfamily='Times New Roman', fontsize=16)
ax.set_xticks(np.arange(0, 25, 2))
ax.set_xticklabels([f'{int(x)}' for x in np.arange(0, 25, 2)], fontfamily='Times New Roman',fontsize=size-2)

# Set y-axis tick positions and labels manually.
y_ticks = np.arange(sol_range[0]+87, sol_range[1] + 1, 100)  # Set a tick every 100 sols.
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks.astype(int), fontfamily='Times New Roman', fontsize=size-2)

# Add a colorbar.
norm = colors.Normalize(vmin=0, vmax=16)
# cmap=plt.get_cmap('viridis')
cmap=plt.get_cmap('plasma')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# divider = make_axes_locatable(ax)
sm.set_array(wind_speed)
cbar = fig.colorbar(sm, cax=ax.inset_axes((0., -0.12, 0.4, 0.02)),orientation='horizontal',extend='both')
plt.text(0.10,-0.18,'Wind Speed (m/s)',transform=ax.transAxes,fontsize=12)

'''
cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label('Wind Speed (m/s)', fontfamily='Times New Roman', fontsize=size)
cbar.ax.tick_params(labelsize=size-2)
'''
plt.tight_layout()
plt.show()