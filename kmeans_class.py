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
import glob

from osgeo import gdal
from osgeo import ogr
import numpy as np
import numpy.ma as ma
from rios import cuiprogress
from rios import calcstats
from rios import rat
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter


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
    p.add_argument("--regionuid", required=True, help=("Region name for output files"))
    p.add_argument("--noClasses", default=8, help=("desired number of output classes in classification image"))
    p.add_argument("--median", default=0, type=int, help=("size of median filter window to apply to output classifications"))
        
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
    for feature in layer:
        geom = feature.GetGeometryRef()
        identifier ="_".join(feature.GetField(uid).split(" "))
        outshape = shapebasename + identifier + '.shp'
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
    drvr = gdal.GetDriverByName('GTiff')
    ds = drvr.Create(outfile, n, m, nBands, dType, ['COMPRESS=LZW'])
    band = ds.GetRasterBand(1)
    
    
    band.WriteArray(inputArray)
    ds.SetGeoTransform((TLX, c, 0, TLY, 0, -c))
    ds.SetProjection(proj)
    progress=cuiprogress.CUIProgressBar() 
    calcstats.calcStats(ds, progress, ignore=nulVal)
    del ds
    
    
def doNDVI(infile, redBand, NIRBand, outfileNDVI):
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
        nulVal = 0
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
    nulVal = 0
    proj = ds.GetProjection() 
    print("Doing kmeans %s" %infile)
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


def doSubset(indexFile, shapefile, noClasses):
    """
    Subset the index file using the input shapefile
    Use gdal_rasterize to create output ndvi/gndvi image for each shapefile
    """    
    
    #Check if shapefiles in index image extent
    ds = gdal.Open(indexFile)
    geotransform = ds.GetGeoTransform()
    m = ds.RasterXSize
    n = ds.RasterYSize
    TLX = geotransform[0]
    TLY = geotransform[3]
    c = geotransform[1]
    BRX = float(TLX) + (float(c) * float(m))
    BRY = float(TLY) - (float(c) * float(n))
    
    
    #check if shape in image extent - if it is subset and do isodata classification
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataSource = driver.Open(shapefile, 0)
    layer = dataSource.GetLayer()
    extent = layer.GetExtent()
    xmin = np.floor(float(extent[0])) 
    ymin = np.floor(float(extent[2])) 
    xmax = np.ceil(float(extent[1]))
    ymax = np.ceil(float(extent[3])) 
    if xmin > TLX and xmax < BRX and ymin > BRY and ymax < TLY:
        #need to make a fresh copy of the NDVI raster to burn into - now subseting the main raster to each polygon extent
        subsetRaster = "%s_%s_subset.tif" %(indexFile.split('.')[0],shapefile.split('.')[0])
        if not os.path.exists(subsetRaster):
            print("Doing subset %s" %(subsetRaster))
            cmd = "gdal_translate -projwin %s %s %s %s -a_nodata 0 %s %s" %(xmin,ymax,xmax,ymin,indexFile,subsetRaster)
            os.system(cmd)
            cmd = "gdal_rasterize -i -burn 0 -l %s %s %s" %(shapefile.split('.')[0], shapefile, subsetRaster)
            os.system(cmd)

        #do the kmeans classification
        kmeansRaster = doKMeans(subsetRaster, noClasses)
    else:
        print("Shapefile %s not within imagery extent" %shapefile)
    os.remove(subsetRaster)
    return kmeansRaster             
       
def creatColorTable():
    
    colorArray = np.array([
        [0,240,240,240,0],
        [1,255,0,0,0],
        [2,255,165,0,1],
        [3,255,255,0,1],
        [4,0,255,0,1],
        [5,0,255,255,1],
        [6,0,0,255,1],
        [7,176,48,96,1],
        [8,255,0,255,1]]
    )            

    return colorArray
    
def medianFilter(infile, nullVal, winsize, outfile):  
    
    
    ds = gdal.Open(infile)
    
    imageArray = np.array(ds.GetRasterBand(int(1)).ReadAsArray())
    
    m = ds.RasterXSize
    n = ds.RasterYSize

    #Find input image origin (TLX,TLY) and pixel size
    geotransform = ds.GetGeoTransform()
    TLX = geotransform[0]
    TLY = geotransform[3]
    c = geotransform[1]
    nulVal = 0
    proj = ds.GetProjection() 
    
    medianFilterArray = median_filter(imageArray, size=winsize)
    writeOutImg(medianFilterArray, outfile, m, n, c, TLX, TLY, nulVal, proj, gdal.GDT_Byte)
    #Create color table
    colorArray = creatColorTable()
    rat.setColorTable(outfile,colorArray)
    
                    
    
def main():

    #Set up inputs
    cmdargs = getCmdargs()
    infile = cmdargs.infile
    mastershapefile = cmdargs.mastershapefile
    regionuid = cmdargs.regionuid
    uid = cmdargs.uid 
    redBand = cmdargs.redBand
    NIRBand = cmdargs.NIRBand
    noClasses = cmdargs.noClasses
    median = cmdargs.median
    
    #Check the infile and shapefile exist
    if not os.path.exists(infile):
        sys.exit("Specified infile %s does not exisit in local directory" %infile)
    elif not os.path.exists(mastershapefile):    
        sys.exit("Specified shapefile %s does not exisit in local directory" %mastershapefile)
    
    
    #create NDVI (or GNDVI) Image - depending on band input as "redBand"
    outfileNDVI = '%s_NDVI_python.tif' %(infile.split('.')[0])
    if not os.path.exists(outfileNDVI):
        print("Doing NDVI %s" %(outfileNDVI))
        outfileNDVI = doNDVI(infile, redBand, NIRBand, outfileNDVI)
    
    #Subset NDVI/GNDVI using all shapefiles (i.e. one regional subset) and perform kmeans classification    
    print("Doing region kmeans")
    kmeansRasterRegion = doSubset(outfileNDVI, mastershapefile, noClasses)
    
    #Rename the file to something more sensible
    if os.path.exists(kmeansRasterRegion):
        kmeansRasterRegionRename = "%s_%s_KMEANS_REGION.tif" %(infile.split('.')[0], regionuid)
        cmd = "mv %s %s" %(kmeansRasterRegion, kmeansRasterRegionRename)
        os.system(cmd)
        
        #Create color table
        colorArray = creatColorTable()
        rat.setColorTable(kmeansRasterRegionRename,colorArray)
        
        #get rid of the pesky .tif.aux file
        kmeansRasterRegionAuxFile = glob.glob('%s*' %kmeansRasterRegion.split('.')[0])       
        for filex in kmeansRasterRegionAuxFile:
            os.remove(filex)
        
        print("Finished creating region raster: %s" %kmeansRasterRegionRename)
     
    #create single feature shapefiles for each row in the input shapefile    
    outshapeList = mksingleshape(mastershapefile, uid)
            
    #Subset image by each single shapefile and perform kmeans classification
    kmeansRasterList = []
    for outshape in outshapeList:
        kmeansRasterBlock = doSubset(outfileNDVI, outshape, noClasses)
        kmeansRasterList.append(kmeansRasterBlock)                    
    
    #create a mosaic of the outputs
    outmosaic = "%s_%s_KMEANS_BLOCKS.tif" %(infile.split('.')[0], regionuid)
    if not os.path.exists(outmosaic):
        cmd = "gdal_merge.py -o %s -of GTiff -co COMPRESS=LZW -co BIGTIFF=IF_SAFER -ot Byte -pct -n 0 -a_nodata 0 %s"  %(outmosaic, " ".join(kmeansRasterList))
        os.system(cmd)   
        #Create color table
        colorArray = creatColorTable()
        rat.setColorTable(outmosaic,colorArray)
    
    if os.path.exists(outmosaic):    
        print("Finished creating block mosaic: %s" %outmosaic)
        
        try:
            #Tidy up - delete individual block kmeans files and shapefiles
            for kmeansFile in kmeansRasterList:
                kmeansFileAll = glob.glob('%s*' %kmeansFile.split('.')[0])
                for kmeansFileAllFile in kmeansFileAll:
                    os.remove(kmeansFileAllFile)
            for outshape in outshapeList:
                outshapeAll = glob.glob('%s*' %outshape.split('.')[0])
                for outshapeAllFile in outshapeAll:
                    os.remove(outshapeAllFile)
        except:
            print("Can't remove all files for cleanup - they are open in another program")            

    #Apply median filter for imagery with small pixel size i.e. Worldview
    if median > 0:
        kmeansRasterRegionRenameMedian = "%s_MEDIAN_TEST%s.tif" %(kmeansRasterRegionRename.split('.')[0], median)
        outmosaicMedian = "%s_MEDIAN_TEST%s.tif" %(outmosaic.split('.')[0], median)
        if not os.path.exists(kmeansRasterRegionRenameMedian):
            medianFilter(kmeansRasterRegionRename, 0, median, kmeansRasterRegionRenameMedian) 
        if not os.path.exists(outmosaicMedian):
            medianFilter(outmosaic, 0, median, outmosaicMedian)
                       

        
if __name__ == "__main__":
    main()
