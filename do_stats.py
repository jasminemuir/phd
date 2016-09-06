#!/usr/bin/env python
"""
Script for creation of isodata image from input image

Takes an input single band image and outputs the isodata classification of the computed NDVI image
based on subseting from an input shapefile defining subregion extent (one polygon per sub region) - polygons
can be muti-part features.

Isodata classification is computed using the open source pyradar implementation:
http://pyradar-tools.readthedocs.io/en/latest/_modules/pyradar/classifiers/isodata.html
http://pyradar-tools.readthedocs.io/en/latest/examples.html#example-of-isodata

Isodata parameters currently hard-coded in this script - can modify to add to command line.


Written by: Jasmine Muir at University of New England
Date: 1/7/2016

Usage: ndvi_isodata.py <input_image> <input_shapefile> <image_red_band_position> <image_nearinfrared_band_position> 
Example: ./ndvi_isodata.py S15_8March2015_MidNSW_ortho_TOA_LS boundaries_subset_gda_remerge.shp 2 3
"""
from __future__ import print_function
import sys
import os
import math
import random
from osgeo import gdal
from osgeo import ogr
import numpy as np
import numpy.ma as ma
from rios import cuiprogress
from rios import calcstats
from scipy.stats.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

def doCorrelation(infile,outfile):
    f = pd.read_csv(infile)

    fruitNo = (f['Fruit number'])
    weight = (f['Total weight (kg)'])
    #size = (f['Avg Fruit size'])
    print(f)
    
    df = f.iloc[:,15:]
    names = df.columns
    
    outstats = open(outfile,'w')
    outstats.write("ColName,fruitNoR,weightR,fruitNoSig,weightSig\n")

    for i, col in enumerate(df):
        dfcol = df[col]
        namecol = names[i]
        print(namecol)
        try:
            s1 = pearsonr(dfcol, fruitNo)
            s2 = pearsonr(dfcol, weight)
            print(namecol,s1[0],s2[0],s1[1],s2[1])
            outline = "%s,%s,%s,%s,%s\n" %(namecol,s1[0],s2[0],s1[1],s2[1])
            outstats.write(outline)
        except:
            print("can't use column %s" %namecol)    
    plt.plot((f['Red']), fruitNo,'ro')
    plt.show()    
    
    outstats.close()    
        
       
    
def main():
    infile = sys.argv[1]
    print(infile)
    
    outfile = sys.argv[2]
   
    if os.path.exists(outfile):
        os.remove(outfile)
    doCorrelation(infile,outfile)    
    
 
if __name__ == "__main__":
    main()
    
    #outstats.close()
