#!/usr/bin/env python
"""
Script to calculate the mean value for each polygon in the input shapefile from a Worldview Image..

Also calculates the 20 vegetation indicies and outputs to CSV file for further analysis.


Written by: Jasmine Muir at University of New England
Date: 1/7/2016

Usage: mean_poly.py <input_image> <input_shapefile> <out_csv>
Example: ./mean_poly.py S15_8March2015_MidNSW_ortho_TOA_LS boundaries_subset_gda_remerge.shp outstats.csv
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
        #outshape = shapebasename + str(feature.GetField("FID")) + '.shp'
        #block = str(feature.GetField("Block"))
        block = str(feature.GetField("Block")).strip()
        #tree = str(feature.GetField("Name")).strip()
        name = str(feature.GetField("UID")).strip()
        #row = name.split("T")[0]
        #treeNo = "T" + name.split("T")[1]
        #row = tree.split(" ")[0]
        #treeNo = tree.split(" ")[1]
        print(block,"/",name,"/")
        outshape = "Block%s_%s_proj.shp" %(block,name)
        count += 1
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
    
    
def doNDVI(inImage, redBand, NIRBand):
    outfileNDVI = inImage.split('.')[0]+'_NDVI_python.tif'
    if not os.path.exists(outfileNDVI):
        ds = gdal.Open(inImage)
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
        writeOutImg(ndvi, outfileNDVI, m, n, c, TLX, TLY, nulVal, proj, gdal.GDT_Float32)
        
    return outfileNDVI
    
def calcVegInd(shpMeanBandList):
    
    
    #BANDS
    CoastalBlue = shpMeanBandList[0]
    Blue = shpMeanBandList[1]
    Green = shpMeanBandList[2]
    Yellow = shpMeanBandList[3]
    Red = shpMeanBandList[4]
    RedEdge = shpMeanBandList[5]
    NIR1 = shpMeanBandList[6]
    NIR2 = shpMeanBandList[7]
    
    #VEGINDICIES
    RENDVI = (RedEdge-Red)/(RedEdge+Red) 
    N1_RE_NDVI = (NIR1-Red)/(NIR1+RedEdge)
    N1_N2_NDVI = (NIR2-Red)/(NIR2+NIR1)
    TCARI = 3*((RedEdge-Red)-0.2*(RedEdge-Green)*(RedEdge/Red))
    SIPI = (NIR1-Blue)/(NIR1-Red)
    NIR1_GNDVI = (NIR1-Green)/(NIR1+Green)
    MSR = ((NIR1/Red)-1)/(math.sqrt((NIR1/Red))+1)
    NIR1PCD = NIR1/Red
    N1NDVI =(NIR1-Red)/(NIR1+Red)
    N2NDVI = (NIR2-Red)/(NIR2+Red)
    N1RENDVI = (NIR1-RedEdge)/(NIR1+RedEdge)
    N2RENDVI = (NIR2-RedEdge)/(NIR2+RedEdge)
    CB_SIPI = (NIR1-CoastalBlue)/(NIR1+CoastalBlue)
    YellowSAVI = ((NIR1-CoastalBlue)*(1+0.5))/(NIR1+CoastalBlue+0.5)
    RDVI = (NIR1-Red)/(math.sqrt(NIR1+Red))
    RDVI2 = (NIR2-Red)/(math.sqrt(NIR2+Red))
    TDVI1 = math.sqrt(0.5+((NIR1-Red)/(NIR1+Red)))
    TDVI2 = math.sqrt(0.5+((NIR2-Red)/(NIR2+Red)))
    EVI2N1 = (2.5*(NIR1-Red))/(1+NIR1+(2.4*Red))
    EVI2N2 = (2.5*(NIR2-Red))/(1+NIR2+(2.4*Red))
        
    
    vegIn = [RENDVI,N1_RE_NDVI,N1_N2_NDVI,TCARI,SIPI,NIR1_GNDVI,MSR,NIR1PCD,N1NDVI,N2NDVI,N1RENDVI,N2RENDVI,CB_SIPI,YellowSAVI,RDVI,RDVI2,TDVI1,TDVI2,EVI2N1,EVI2N2]
    return vegIn
    
def doCorrelation(outfile):
    f = pd.read_csv(outfile)

    fruitNo = (f['fruit'])
    df = f.iloc[:,3:]
    names = df.columns

    for i, col in enumerate(df):
        dfcol = df[col]
        namecol = names[i]
        s = pearsonr(dfcol, fruitNo)
        print(namecol,s)
        
       
    
def main():
    inImage = sys.argv[1]
    shapefile = sys.argv[2]
    
    outfile = sys.argv[3]
    
    if os.path.exists(outfile):
        os.remove(outfile)
    outstats = open(outfile,'w')
    outstats.write("outshape,NoPix,CoastalBlue,Blue,Green,Yellow,Red,RedEdge,NIR1,NIR2,RENDVI,N1_RE_NDVI,N1_N2_NDVI,TCARI,SIPI,NIR1_GNDVI,MSR,NIR1PCD,N1NDVI,N2NDVI,N1RENDVI,N2RENDVI,CB_SIPI,YellowSAVI,RDVI,RDVI2,TDVI1,TDVI2,EVI2N1,EVI2N2\n") 
    
    
    
    #create single feature shapefiles from the input shapefile    
    outshapeList = mksingleshape(shapefile)
    
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
    
    string = ""
    for band in range(nBands):
        string = string + "-b %s " %(int(band)+1)
    print(string)    
            
    #use gdal_rasterize to create output ndvi image for each single feature shapefile
    outList = []
    for outshape in outshapeList:
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
            #need to make a fresh copy of the NDVI raster to burn into - now subseting the main raster to each polygon extent
            outraster = inImage.split('.')[0] + '_' + outshape.split('.')[0] + '.tif'
            if os.path.exists(outraster):
                os.remove(outraster)
            if not os.path.exists(outraster):
                cmd = "gdal_translate -projwin %s %s %s %s -a_nodata -1 %s %s" %(xmin,ymax,xmax,ymin,inImage,outraster)
                #cmd = "cp %s %s" %(outfileNDVI, outraster)
                os.system(cmd)

                cmd = "gdal_rasterize %s -i -burn -1 -l %s %s %s" %(string, outshape.split('.')[0], outshape, outraster)
                print
                os.system(cmd)
                print('2')
                
                dsShp = gdal.Open(outraster)
                shpMeanBandList = []
                for i in range(nBands):
                    bandData = dsShp.GetRasterBand(i+1)
                    bandArray = bandData.ReadAsArray()
                    maskedBandArray = np.ma.masked_where(bandArray==-1,bandArray)
                    #print(maskedBandArray)
                    mean = np.ma.mean(maskedBandArray)
                    numPix = maskedBandArray.count()
                    shpMeanBandList.append(mean)
                    #print(outshape,i,mean,numPix)
                #print(outshape, numPix, shpMeanBandList)
                vegInd = calcVegInd(shpMeanBandList)
                #fruit = random.randrange(0, 101, 1)
                outline = "%s,%s,%s,%s\n"%(outshape.split('.')[0], str(numPix), ",".join((str(s) for s in shpMeanBandList)), ",".join(str(v) for v in vegInd))
                outstats.write(outline)
                print(outline)
                #print(vegInd)
                       
        else:
            print("Shapefile %s not within imagery extent" %outshape)
    
    outstats.close()
    
    #doCorrelation(outfile)         
                
    

        
if __name__ == "__main__":
    main()
