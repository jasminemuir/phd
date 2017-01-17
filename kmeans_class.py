#!/usr/bin/env python
"""
Script for creation of kmeans classification image from input image

Takes an input multi-band image and outputs the kmeans classification of the computed NDVI/GNDVI image
based on subseting from an input shapefile defining subregion extent (one polygon per sub region).
Subseting is completed based on each feature/row. Th can be muti-part features.


Written by: Jasmine Muir at University of New England
Date: 1/7/2016

Usage: ndvi_isodata.py <input_image> <input_shapefile> <image_red_band_position> <image_nearinfrared_band_position> 
Example: ./ndvi_isodata.py S15_8March2015_MidNSW_ortho_TOA_LS boundaries_subset_gda_remerge.shp 2 3
"""
from __future__ import print_function
import sys
import os
import argparse

from osgeo import gdal
from osgeo import ogr
import numpy as np
import numpy.ma as ma
from rios import cuiprogress
from rios import calcstats
from sklearn.cluster import KMeans


def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--infile", required=True, help=("Input multispectral image"))
    p.add_argument("--redBand", required=True, help=("Red band (or Green for GNVI) for NDVI calculation"))
    p.add_argument("--NIRBand", required=True, help=("NIR band for NDVI calculation"))
    p.add_argument("-s", "--mastershapefile", required=True, help="Shapefile to subset region. Each feature is"+
        "one subset. Need multipart polygons to do multiple areas in a single subset (i.e. region). Needs unique id field")
    p.add_argument("--uid", default='ID', help=("Field with unique ID in shapefile"))
    p.add_argument("--noClasses", default=8, help=("desired number of output classes in classification image"))
    p.add_argument("-o","--outmosaic", required=True, help=("name of output mosaic"))           
        
    cmdargs = p.parse_args()
    
    return cmdargs


def mksingleshape(shapefile, uid):
    """
    Takes an input shapefile and creates a new shapefile for each feature
    """
    
    shapebasename = "_".join(shapefile.split('.')[0].split(" "))
    outshapeList = []
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    proj = layer.GetSpatialRef()
    print("shapefile", proj)
    for feature in layer:
        geom = feature.GetGeometryRef()
        identifier ="_".join(feature.GetField(uid).split(" "))
        outshape = shapebasename + identifier + '.shp'
        print(outshape)
        outshapeList.append(outshape)
        print(outshape)
        if not os.path.exists(outshape):
            # Create the output shapefile
            outDataSource = driver.CreateDataSource(outshape)
            outLayer = outDataSource.CreateLayer(outshape.split('.')[0], proj, geom_type=ogr.wkbPolygon)      
            outLayer.CreateFeature(feature)
            outDataSource.Destroy()
    dataSource.Destroy()
          
    return outshapeList
    

def createColorTable():   
    ct = gdal.ColorTable()
    for i in range(256):
        ct.SetColorEntry( i, (255, 255 - i, i, 255) )
    return ct    
    

def writeOutImg(inputArray, outfile, n, m, c, TLX, TLY, nulVal, proj, dType):
    # Write the output DEM into an image file with GDAL
    nBands = 1  
    drvr = gdal.GetDriverByName('GTiff')
    ds = drvr.Create(outfile, n, m, nBands, dType, ['COMPRESS=LZW'])
    band = ds.GetRasterBand(1)
    
  #  if dType is gdal.GDT_Byte:
 #       ct = createColorTable()
#        band.SetRasterColorTable(ct)
    
    band.WriteArray(inputArray)
    ds.SetGeoTransform((TLX, c, 0, TLY, 0, -c))
    ds.SetProjection(proj)
    progress=cuiprogress.CUIProgressBar() 
    calcstats.calcStats(ds, progress, ignore=nulVal)
    del ds
    
    
def doNDVI(infile, redBand, NIRBand):
    outfileNDVI = infile.split('.')[0]+'_NDVI_python.tif'
    if not os.path.exists(outfileNDVI):
        ds = gdal.Open(infile)
        #Find row and column number
        m = ds.RasterXSize
        n = ds.RasterYSize
        #Find input image origin (TLX,TLY) and pixel size
        geotransform = ds.GetGeoTransform()
        TLX = geotransform[0]
        TLY = geotransform[3]
        c = geotransform[1] 
        proj = ds.GetProjection()
        print(proj)   
        nulVal = -1    
        print(outfileNDVI, n, m, c, TLX, TLY,nulVal)
        red = np.array(ds.GetRasterBand(int(redBand)).ReadAsArray())
        nir = np.array(ds.GetRasterBand(int(NIRBand)).ReadAsArray())
        ndvi = (nir-red)/(nir+red)
        ndvi = np.where(np.isnan(ndvi),nulVal, ndvi)
        writeOutImg(ndvi, outfileNDVI, m, n, c, TLX, TLY, nulVal, proj, gdal.GDT_Float32)
        
    return outfileNDVI
    
def doKMeans(infile, numClusters):
    ds = gdal.Open(infile)
    
    imageArray = np.array(ds.GetRasterBand(int(1)).ReadAsArray())
    
    m = ds.RasterXSize
    n = ds.RasterYSize

    #Find input image origin (TLX,TLY) and pixel size
    geotransform = ds.GetGeoTransform()
    TLX = geotransform[0]
    TLY = geotransform[3]
    c = geotransform[1]
    nulVal = -1
    proj = ds.GetProjection() 

    imageArray1D = np.column_stack([imageArray.flatten()])
    k_means = KMeans(n_clusters=int(numClusters)+1)    #Add 1 to number of clusters to account for no data class
    
    pred = k_means.fit_predict(imageArray1D)
    
    clusterCentres = k_means.cluster_centers_.flatten()
    
    from_values = np.argsort(clusterCentres)
    to_values = np.arange(from_values.size)
    sort_idx = np.argsort(from_values)
    idx = np.searchsorted(from_values, pred, sorter = sort_idx)
    out = to_values[sort_idx][idx]
    
  
    classImageArray = out.reshape(imageArray.shape)
    
    outfileClass = "%s_kmeans.tif" %infile.split('.')[0]
            
    writeOutImg(classImageArray, outfileClass, m, n, c, TLX, TLY, nulVal, proj, gdal.GDT_Byte)

    del ds
    return outfileClass    
              
       
    
def main():

    #Set up inputs
    cmdargs = getCmdargs()
    print(cmdargs)
    infile = cmdargs.infile
    mastershapefile = cmdargs.mastershapefile
    uid = cmdargs.uid 
    redBand = cmdargs.redBand
    NIRBand = cmdargs.NIRBand
    noClasses = cmdargs.noClasses
    outmosaic = cmdargs.outmosaic
    
    #create NDVI (or GNDVI) Image
    outfileNDVI = doNDVI(infile, redBand, NIRBand)
    
    #create single feature shapefiles from the input shapefile    
    outshapeList = mksingleshape(mastershapefile, uid)
    
    #Check if shapefiles in image extent
    ds = gdal.Open(infile)
    geotransform = ds.GetGeoTransform()
    m = ds.RasterXSize
    n = ds.RasterYSize
    TLX = geotransform[0]
    TLY = geotransform[3]
    c = geotransform[1]
    BRX = float(TLX) + (float(c) * float(m))
    BRY = float(TLY) - (float(c) * float(n))
            
    #use gdal_rasterize to create output ndvi image for each single feature shapefile
    outfileCLassList = []
    for outshape in outshapeList:
        #check if shape in image extent - if it is subset and do isodata classification
        driver = ogr.GetDriverByName("ESRI Shapefile")
        dataSource = driver.Open(outshape, 0)
        layer = dataSource.GetLayer()
        extent = layer.GetExtent()
        xmin = np.floor(float(extent[0])) 
        ymin = np.floor(float(extent[2])) 
        xmax = np.ceil(float(extent[1]))
        ymax = np.ceil(float(extent[3])) 
        if xmin > TLX and xmax < BRX and ymin > BRY and ymax < TLY:
            #need to make a fresh copy of the NDVI raster to burn into - now subseting the main raster to each polygon extent
            outraster = outfileNDVI.split('.')[0] + '_' + outshape.split('.')[0] + '.tif'
            if not os.path.exists(outraster):
                cmd = "gdal_translate -projwin %s %s %s %s -a_nodata -1 %s %s" %(xmin,ymax,xmax,ymin,outfileNDVI,outraster)
                #cmd = "cp %s %s" %(outfileNDVI, outraster)
                os.system(cmd)

                cmd = "gdal_rasterize -i -burn -1 -l %s %s %s" %(outshape.split('.')[0], outshape, outraster)
                print
                os.system(cmd)
                print('2')

            #do the kmeans classification
            outfileClass = doKMeans(outraster, noClasses)
            outfileCLassList.append(outfileClass)
            cmd = "rm %s* %s*" %(outraster, outshape.split('.')[0])
            #os.system(cmd)

            
            
        else:
            print("Shapefile %s not within imagery extent" %outshape)
                
    
    #create a mosaic of the outputs
    cmd = "gdal_merge.py -o %s -of GTiff -co COMPRESS=LZW -co BIGTIFF=IF_SAFER -ot Byte -pct -n 0 -a_nodata 0 %s"  %(outmosaic, " ".join(outfileCLassList))
    os.system(cmd)
    if os.path.exists(outmosaic):    
        print("finished creating mosaic %s" %outmosaic)
    
    cmd = "rm %s" %(" ".join(outfileCLassList))
    #os.system(cmd) 
        
if __name__ == "__main__":
    main()
