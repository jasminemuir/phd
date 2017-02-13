#!/usr/bin/env python

import sys, os
import numpy
from rios import applier
from rios import fileinfo
import matplotlib.mlab as ml
from pylidar.toolbox import interpolation
import pynninterp
from osgeo import gdal
from numba import jit
import math
from scipy.ndimage import median_filter

###Takes the filtered minz images - not the interpolated ones
###i.e. /media/cdrive/scripts/create_minimage_jit.py 170209_100610_ScanPos002_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_104505_ScanPos004_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_110117_ScanPos007_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_111913_ScanPos010_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_113732_ScanPos012_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_115557_ScanPos014_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_121113_ScanPos016_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_133649_ScanPos103_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_134605_ScanPos104_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_135521_ScanPos105_20cm_none_minz50m_median100filter_pmf_globalper.img 170209_140205_ScanPos106_20cm_none_minz50m_median100filter_pmf_globalper.img


def saveImage(outfile, imgArr, xmin, ymax, res, proj, nullVal):
    """
    Save the given image array as a GDAL raster file. Only saves a single band. 
    """
    
    print("Saving image %s" %(outfile))
    
    gdalTypeDict = {numpy.dtype(numpy.float32):gdal.GDT_Float32,
        numpy.dtype(numpy.float64):gdal.GDT_Float64,
        numpy.dtype(numpy.uint8):gdal.GDT_Byte}
    gdalType = gdalTypeDict[imgArr.dtype]
        
    
    drvr = gdal.GetDriverByName('HFA')
    
    if len(imgArr.shape) == 3:
        (nRows, nCols) = (imgArr.shape[1],imgArr.shape[2])
        nbands = imgArr.shape[0]
        ds = drvr.Create(outfile, nCols, nRows, nbands, gdalType, ['COMPRESS=YES'])
        for i in range(nbands):
            band = ds.GetRasterBand(i+1)
            band.WriteArray(imgArr[i])
            band.SetNoDataValue(nullVal)
    elif len(imgArr.shape) == 2:
        (nRows, nCols) = (imgArr.shape[0],imgArr.shape[1])
        ds = drvr.Create(outfile, nCols, nRows, 1, gdalType, ['COMPRESS=YES'])
        band = ds.GetRasterBand(1)
        band.WriteArray(imgArr)
        band.SetNoDataValue(nullVal)
    #print("proj", proj)      
    if proj != None and len(proj)>50:
        ds.SetProjection(proj)
    ds.SetGeoTransform((xmin, res, 0, ymax, 0, -res))

def medianFilter(zArr, zNull, resolution, mediantype):  
    if mediantype == 'median100':
        winsize = int(2.0/resolution)
        #Windows size needs to be at least 3 and should be an odd number
        if winsize < 3:
            winsize = 3
        elif winsize % 2 == 0:
            #make odd
            winsize=winsize + 1
    else:
        winsize = int(mediantype[6:])        
    #print("median filter type %s and size = %s" %(mediantype, winsize))   s
    zArrMedian = median_filter(zArr, size=winsize) # Median window size of 3x3, but perhaps 5x5 would be better? Not sure.....
    medianDiff = (zArr - zArrMedian)
    # Knock out pixels which are too high above the median
    diffThresh = 0.2   # Just a guess...... this means 2 metres above the surrounding pixels
    tooHigh = (medianDiff > diffThresh)
    zArr[tooHigh] = zNull
    
    return zArr


def doInterp(minfilteredimage, outinterpimage, controlPtArray):
    
    inImage = gdal.Open(minfilteredimage)
    nCols = inImage.RasterXSize
    nRows = inImage.RasterYSize
    geotransform = inImage.GetGeoTransform()
    resolution = geotransform[1]
    xmin = numpy.floor(geotransform[0])
    xmax = numpy.ceil((xmin+(nCols*resolution))+resolution)
    ymax = numpy.ceil(geotransform[3])
    ymin = numpy.floor((ymax-(nRows*resolution))-resolution)
    
    
    print(xmin,xmax,ymin,ymax, resolution)
    print(geotransform)
    

    proj = inImage.GetProjection()
    nulVal = inImage.GetRasterBand(1).GetNoDataValue()
    
    
    

    z = ((inImage.GetRasterBand(1)).ReadAsArray())
    x = ((inImage.GetRasterBand(2)).ReadAsArray())
    y = ((inImage.GetRasterBand(3)).ReadAsArray())
    
    filteredOutMask = (z==nulVal)

    mz = numpy.ma.array(z, mask=filteredOutMask).compressed()
    mx = numpy.ma.array(x, mask=filteredOutMask).compressed()
    my = numpy.ma.array(y, mask=filteredOutMask).compressed()

    # Generate a regular grid to interpolate the data.
    xi = numpy.linspace(xmin, xmax-resolution, nCols)
    yi = numpy.linspace(ymin, ymax-resolution, nRows)
    xi, yi = numpy.meshgrid(xi, yi)


#        ## Interpolate the values of z for all points in the rectangular grid
#        ## Method 1 - Interpolate by matplotlib delaunay triangularizatio and nearest neigh. PLEASE NOTE! THIS FAILS QUITE OFTEN (http://matplotlib.org/api/mlab_api.html#matplotlib.mlab.griddata) But there might be a solution - install mpl_toolkits.natgrid (http://matplotlib.org/mpl_toolkits/)
#        #zi = ml.griddata(mx,my,mz,xi,yi,interp='nn') #interpolation is 'nn' by default (natural neighbour based on delaunay triangulation) but 'linear' is faster (see http://matplotlib.1069221.n5.nabble.com/speeding-up-griddata-td20906.html)
    #surfaceImg = numpy.flipud(zi)

    #use the pylidar natural neighbour interpolation routine


    surfaceImg = interpolation.interpGrid(mx, my, mz, (xi,yi), 'pynn')
    surfaceImg = numpy.flipud(surfaceImg)
    #elif cmdargs.interp == 'mann':
    #zi = ml.griddata(mx,my,mz,xi,yi,interp='nn') #interpolation is 'nn' by default (natural neighbour based on delaunay triangulation) but 'linear' is faster (see http://matplotlib.1069221.n5.nabble.com/speeding-up-griddata-td20906.html)
    #surfaceImg = numpy.flipud(zi)     


    saveImage(outinterpimage, surfaceImg, xmin, ymax, resolution, proj, nulVal)
    
    xyAtControl = controlPtArray[:, :2]
    zAtControl = controlPtArray[:, 2]
    interpzAtControl = pynninterp.NaturalNeighbourPts(mx.astype(numpy.float64), my.astype(numpy.float64), mz.astype(numpy.float64), xyAtControl)
    zDiff = zAtControl - interpzAtControl
    outf = open('controlCompare_noMed_20151125.csv', 'w')
    for i in range(len(zAtControl)):
        outf.write("{},{},{}\n".format(interpzAtControl[i], zAtControl[i], zDiff[i]))
    outf.close()



def doMinimum(info, inputs, outputs, otherargs):
    "Called from RIOS. Average the input files"

    zstack = numpy.array([imgs[0] for imgs in inputs.imgs])
    #zstack = numpy.array([imgxyz[0] for imgxyz in inputs.minxyz])
    
    #Make the 0 no data values equal to 1000 - so they don't get confused with the min in argmin
    zstack = numpy.where(zstack!=0,zstack,1000)
    xstack = numpy.array([imgxyz[1] for imgxyz in inputs.minxyz])
    ystack = numpy.array([imgxyz[2] for imgxyz in inputs.minxyz])
    
   
    layerWithMin = numpy.argmin(zstack, axis=0)
    
    minZ = selectLayerPerPixel(zstack, layerWithMin)    
    xAtMin = selectLayerPerPixel(xstack, layerWithMin)
    yAtMin = selectLayerPerPixel(ystack, layerWithMin)
    
    #deal with no data - do minz last as otherwise the replace is done before it's used in the x and y arrays   
    xAtMin = numpy.where(minZ==1000,0,xAtMin)
    yAtMin = numpy.where(minZ==1000,0,yAtMin)
    minZ = numpy.where(minZ==1000,0,minZ)
    

    outputs.min = numpy.array([minZ, xAtMin, yAtMin])
    
    


@jit
def selectLayerPerPixel(arr, lyrNdx):
    """
    From the 3-d arr, select on the first index (layer number), using a 2-d array
    of index values (lyrNdx). The shape of lyrNdx must match the shape of the 2nd and 3rd
    indexes of arr. 
    
    The 2nd and 3rd indexes represent rows and columns of an image stack, and for 
    each pixel we are selecting a different layer. 
    
    Return a 2-d array, representing a single image. 
    
    """
    (nBands, nRows, nCols) = arr.shape
    outImg = numpy.zeros((nRows, nCols), dtype=arr.dtype)
    for r in range(nRows):
        for c in range(nCols):           
            lyr = lyrNdx[r, c]        
            outImg[r, c] = arr[lyr, r, c]
    return outImg



infiles = applier.FilenameAssociations()
# names of imput images
controlPtLines = [line.strip().split(',') for line in open(sys.argv[1])]
controlPtXYZ = [[float(v) for v in line[1:4]] for line in controlPtLines]
controlPtArray = numpy.array(controlPtXYZ)


infiles.imgs = sys.argv[2:]
print(infiles.imgs)
#create list of minz files using filtered input names i.e. relies on naming convention
infiles.minxyz = []
for fil in infiles.imgs:
    minxyz = "%s.img" %("_".join(fil.split('_')[:6]))
    infiles.minxyz.append(minxyz)
    

otherargs = applier.OtherInputs()
otherargs.noDataVal = float(fileinfo.ImageInfo(infiles.imgs[0]).nodataval[0])
print(otherargs.noDataVal)

# Last name given is the output
outfiles = applier.FilenameAssociations()
outfiles.min = "AA_jit_MedUprightpos22cm_combinez%s" %("_".join(sys.argv[-1].split("_")[3:]))
if os.path.exists(outfiles.min):
    os.remove(outfiles.min)
controls = applier.ApplierControls()
controls.setFootprintType(applier.UNION)
applier.apply(doMinimum, infiles, outfiles, otherargs, controls=controls)
inImage = gdal.Open(outfiles.min)
nulVal = inImage.GetRasterBand(1).GetNoDataValue()
minZ = (inImage.GetRasterBand(1)).ReadAsArray()
geotransform = inImage.GetGeoTransform()
resolution = geotransform[1]

#medianZ = medianFilter(minZ, nulVal, resolution, 'median100')

outinterpimage = "AA_jit_MedUprightpos2_2cm_interpz%s" %("_".join(sys.argv[-1].split("_")[3:]))
if os.path.exists(outinterpimage):
    os.remove(outinterpimage)

#doInterp(outfiles.min, medianZ, outinterpimage)
doInterp(outfiles.min, outinterpimage, controlPtArray)


