#!/usr/bin/env python
"""
Created on Mon Aug 01 19:12:00 2016

Script to plot correlation values from spectral data
It plots the results for all groups(blocks), as well as for each group(block)

You will need to so some modifications - based on the attributes you want to correlate to each other i.e fruit number.
You will also need to adjust the columns which are used by modifying the index in the script.
If we can standardise the method and field data collection then we won't need to do this anymore.

To use the script specify the input spectral data file and the name of the "grouping" column i.e.
./stats_avo.py Renmark_Avos_Merged_Final_Spectral.csv Block

@author: Jasmine Muir
"""
from __future__ import print_function

import pandas as pd
from pandas.stats.api import ols
import os
import sys
import math
import numpy as np
import statsmodels.formula.api as sm
import statsmodels.api as sm
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as PathEffects


#Input data csv file with spectral info (needs to have block id
filepath =sys.argv[1]

#Column name for grouping
ID = sys.argv[2]
nBlocks = int(sys.argv[3])

f = pd.read_csv(filepath,sep =",")


number = f['Fruit number']
weight = f['Total weight (kg)']
size = f['Avefruitweight']
#print(size)


s=f.groupby(ID)

iteration = 0

fig, axes = plt.subplots(nrows=nBlocks+1, ncols=1)

maxList = []

for name,group in s:
    d=group.corr()
    #print(d.columns)
    c = d.iloc[6:9,10:]
    maxWE = np.max(np.absolute(c.iloc[0]))
    maxWEName = np.argmax(np.absolute(c.iloc[0]))
    maxNo = np.max(np.absolute(c.iloc[1]))
    maxNoName = np.argmax(np.absolute(c.iloc[1]))   
    maxSI = np.max(np.absolute(c.iloc[2]))
    maxSIName = np.argmax(np.absolute(c.iloc[2]))
    print(name,maxWE,maxWEName,maxNo,maxNoName,maxSI,maxSIName)
    maxList.append([name,maxWEName,maxNoName,maxSIName])
    im = axes[iteration].matshow(c, vmin=-1, vmax=1)
    axes[iteration].set_title("Block %s" %name, x=-0.1)   
    for (i, j), z in np.ndenumerate(c):
        axes[iteration].text(j, i, '{:0.2f}'.format(z), ha='center', va='center', size=10, path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
            #Used if want a bounding box (instead of halo on text
            ##bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    iteration+=1
allblocks = f.corr()
t = allblocks.iloc[6:9,10:]
maxWE = np.max(np.absolute(t.iloc[0]))
maxWEName = np.argmax(np.absolute(t.iloc[0]))
maxNo = np.max(np.absolute(t.iloc[1]))
maxNoName = np.argmax(np.absolute(t.iloc[1]))
maxSI = np.max(np.absolute(t.iloc[2]))
maxSIName = np.argmax(np.absolute(t.iloc[2]))
print("allblocks",maxNo,maxNoName,maxWE,maxWEName,maxSI,maxSIName)

print(maxList)

im = axes[iteration].matshow(t,vmin=-1, vmax=1)
axes[iteration].set_title("All Blocks", x=-0.1)
for (i, j), z in np.ndenumerate(t):
    axes[iteration].text(j, i, '{:0.2f}'.format(z), ha='center', va='center', size=10, path_effects=[PathEffects.withStroke(linewidth=3,foreground="w")])
        ##bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    
# Set the ticks and ticklabels for all axes
plt.setp(axes, yticks=range(len(c.index)), yticklabels=c.index)
plt.setp(axes, xticks=range(len(c.columns)), xticklabels="")
plt.sca(axes[0])
plt.xticks(range(len(c.columns)),list(c),rotation='vertical') 
#plt.tight_layout()  

fig.colorbar(im, ax=axes.ravel().tolist())
#plt.show()

#Now plot the highest correlation for each group and factor
box = dict(facecolor='white', pad=5, alpha=0.2)
fig2, axes2 = plt.subplots(nrows=nBlocks+1, ncols=3)

count = 0

for name,group in s:
    x1int = (np.max(group[maxList[count][1]])-np.min(group[maxList[count][1]]))/10.0
    axes2[count][0].set_xlim(np.min(group[maxList[count][1]])-x1int,np.max(group[maxList[count][1]])+x1int)
    axes2[count][0].scatter(group[maxList[count][1]],group['Total weight (kg)'])
    axes2[count][0].set_xlabel(maxList[count][1])
    axes2[count][0].set_ylabel("Block %s" %name, size=12, bbox=box)
    axes2[count][0].yaxis.set_label_coords(-0.3, 0.5)
    x2int = (np.max(group[maxList[count][2]])-np.min(group[maxList[count][2]]))/10.0
    axes2[count][1].set_xlim(np.min(group[maxList[count][2]])-x2int,np.max(group[maxList[count][2]])+x2int)    
    axes2[count][1].scatter(group[maxList[count][2]],group['Fruit number'])
    axes2[count][1].set_xlabel(maxList[count][2])
    x3int = (np.max(group[maxList[count][3]])-np.min(group[maxList[count][3]]))/10.0
    axes2[count][2].set_xlim(np.min(group[maxList[count][3]])-x3int,np.max(group[maxList[count][3]])+x3int)
    axes2[count][2].scatter(group[maxList[count][3]],group['Avefruitweight'])
    axes2[count][2].set_xlabel(maxList[count][3])
    count += 1

#Add the plots for all blocks
x1intAll = (np.max(f[maxNoName])-np.min(f[maxNoName]))/10.0
axes2[count][0].set_xlim(np.min(f[maxNoName])-x1intAll,np.max(f[maxNoName])+x1intAll)
axes2[count][0].scatter(f[maxNoName],f['Total weight (kg)'])
axes2[count][0].set_xlabel(maxNoName)
x2intAll = (np.max(f[maxWEName])-np.min(f[maxWEName]))/10.0
axes2[count][1].set_xlim(np.min(f[maxWEName])-x2intAll,np.max(f[maxWEName])+x2intAll)
axes2[count][1].scatter(f[maxWEName],f['Fruit number'])
axes2[count][1].set_xlabel(maxWEName)
x3intAll = (np.max(f[maxSIName])-np.min(f[maxSIName]))/10.0
axes2[count][2].set_xlim(np.min(f[maxSIName])-x3intAll,np.max(f[maxSIName])+x3intAll)
axes2[count][2].scatter(f[maxSIName],f['Avefruitweight'])
axes2[count][2].set_xlabel(maxSIName)

axes2[0][0].set_title("Fruit Number")
axes2[0][1].set_title("Total Weight (kg)")
axes2[0][2].set_title("Average Fruit Weight (kg)")
axes2[count][0].set_ylabel("All Blocks", size=12,bbox=box)
axes2[count][0].yaxis.set_label_coords(-0.3, 0.5)


plt.suptitle('Plots of Factors with Highest Correlation', size=16)

#plt.tight_layout()  
plt.show()
