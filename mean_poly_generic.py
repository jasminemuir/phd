#!/usr/bin/env python
"""
Script to calculate the mean value for each band from a generic input image, for each polygon in an input shapefile

Does the same kind of thing as Starspan - but only includes pixels whose centre co-ordinate is within the shapefile

Will later modify to include other stats if necessary i.e. standard deviation, median etc.


Written by: Jasmine Muir at University of New England
Date: /7/2016

Usage: mean_poly_generic.py <input_image> <input_shapefile> <out_csv>
Example: ./mean_poly.py S15_8March2015_MidNSW_ortho_TOA_LS boundaries_subset_gda_remerge.shp outstats.csv
"""
from __future__ import print_function, division
import sys
import os
import math
import random
from osgeo import gdal
from osgeo import ogr
import numpy as np
import numpy.ma as ma


def mksingleshape(shapefile):
    shapebasename = shapefile.split('.')[0]
    #Takes an input shapefile and creates a new shapefile for each feature
    outshapeList = []
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    proj = layer.GetSpatialRef()
    count = 1
    for feature in layer:
        geom = feature.GetGeometryRef()
        name = str(feature.GetField("Id")).strip()
        outshape = "%s_out%s.shp" %(shapebasename,name)
        count += 1
        outshapeList.append(outshape)
        if not os.path.exists(outshape):
            # Create the output shapefile
            outDataSource = driver.CreateDataSource(outshape)
            outLayer = outDataSource.CreateLayer(outshape.split('.')[0], proj, geom_type=ogr.wkbPolygon)      
            outLayer.CreateFeature(feature)
            outDataSource.Destroy()
    dataSource.Destroy()
          
    return outshapeList



def writeOutImg(inputArray, outfile, n, m, c, TLX, TLY, nulVal, proj, dType):
    # Write the output DEM into an image file with GDAL
    nBands = 1  
    drvr = gdal.GetDriverByName('HFA')
    ds = drvr.Create(outfile, n, m, nBands, dType, ['COMPRESS=YES'])
    band = ds.GetRasterBand(1)

    band.WriteArray(inputArray)
    ds.SetGeoTransform((TLX, c, 0, TLY, 0, -c))
    ds.SetProjection(proj)
    progress=cuiprogress.CUIProgressBar() 
    calcstats.calcStats(ds, progress, ignore=nulVal)
    del ds
    
      
    
def main():
    inImage = sys.argv[1]
    shapefile = sys.argv[2]
    
    outfile = sys.argv[3]
    
    if os.path.exists(outfile):
        os.remove(outfile)
    outstats = open(outfile,'w')
     
    
    #create single feature shapefiles from the input shapefile    
    (outshapeList) = mksingleshape(shapefile)
    print(outshapeList)
    
    #Check if shapefiles in image extent
    ds = gdal.Open(inImage)
    geotransform = ds.GetGeoTransform()
    m = ds.RasterXSize
    n = ds.RasterYSize
    TLX = geotransform[0]
    TLY = geotransform[3]
    c = geotransform[1]
    BRX = float(TLX) + (float(c) * float(m))
    BRY = float(TLY) - (float(c) * float(n))
    nBands = ds.RasterCount
    
    #Create a header list with labels for each of the bands in the image
    bandHead = []
    for i in range(nBands):
        j = "Band_%s"%(i+1)
        bandHead.append(j)
        
    #Write the output header    
    outstats.write("outshape,NoPix,%s\n"%",".join(bandHead))
    
    #Generate a list of the bands for input to the gdal_rasterize command
    string = ""
    for band in range(nBands):
        string = string + "-b %s " %(int(band)+1)
       
    #use gdal_rasterize to create output image for each single feature shapefile
    outList = []
    for index, outshape in enumerate(outshapeList, start=0):
        #check if shape in image extent - if it is subset and do isodata classification
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(outshape, 0)
        layer = dataSource.GetLayer()
        extent = layer.GetExtent()
        xmin = np.floor(float(extent[0]))
        ymin = np.floor(float(extent[2])) - 2*c 
        xmax = np.ceil(float(extent[1])) + 2*c
        ymax = np.ceil(float(extent[3]))
        if xmin > TLX and xmax < BRX and ymin > BRY and ymax < TLY:
            #need to make a fresh copy of the raster to burn into - now subseting the main raster to each polygon extent
            outraster = inImage.split('.')[0] + '_' + outshape.split('.')[0] + '.tif'
            if os.path.exists(outraster):
                os.remove(outraster)
            if not os.path.exists(outraster):
                cmd = "gdal_translate -projwin %s %s %s %s -a_nodata -1 %s %s" %(xmin,ymax,xmax,ymin,inImage,outraster)
                #cmd = "cp %s %s" %(outfileNDVI, outraster)
                os.system(cmd)

                cmd = "gdal_rasterize %s -i -burn -1 -l %s %s %s" %(string, outshape.split('.')[0], outshape, outraster)
                os.system(cmd)
                
                dsShp = gdal.Open(outraster)
                shpMeanBandList = []
                for i in range(nBands):
                    bandData = dsShp.GetRasterBand(i+1)
                    bandArray = bandData.ReadAsArray()
                    maskedBandArray = np.ma.masked_where(bandArray==-1,bandArray)
                    mean = np.ma.mean(maskedBandArray)
                    numPix = maskedBandArray.count()
                    shpMeanBandList.append(mean)
                outline = "%s,%s,%s\n"%(outshape.split('.')[0], str(numPix),",".join((str(s) for s in shpMeanBandList)))
                outstats.write(outline)
                print(outline)                      
        else:
            print("Shapefile %s not within imagery extent" %outshape)
    
    outstats.close()
        
if __name__ == "__main__":
    main()
