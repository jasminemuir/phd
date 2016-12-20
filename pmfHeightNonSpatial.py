#!/usr/bin/env python
"""
Re-write of Jasmine's PMF/height algorithm. 

Takes an input SPD V4 TLS file. Uses the PMF algorithm to create a ground surface image,
which it holds in memory. Uses this to classify all points as ground or not, and then
adds a HEIGHT field with height above this ground surface. 

Modifies the SPD file in place, changing the CLASSIFICATION and HEIGHT fields. 

Has options to explicitly set the region of calculation with xmin/max, ymin/max. 
Options to save the surface images, either the initial minz image, and/or the filtered
ground surface image. 

"""

from __future__ import print_function, division

import argparse
import os
import sys
import json

import numpy
from numba import jit
from osgeo import gdal
from scipy.ndimage import median_filter
import matplotlib.mlab as ml
import scipy.interpolate as il #for method2, in case the matplotlib griddata method fails

from pylidar import lidarprocessor
from pylidar.lidarformats import generic
from pylidar.toolbox import interpolation

#import pmfneil
import pmfjaz


def getCmdargs():
    """
    Get commandline arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument("infile", help=("Input SPDV4 file. It will be updated with PMF-based "+
        "ground classification, and updated height field on every point"))
    p.add_argument("-r", "--resolution", default=0.1, type=float,
        help="Resolution (i.e. pixel size) (in metres) of DEM image to work with (default=%(default)s)")
    p.add_argument("--blocksize", default=256, type=int,
        help=("Pylidar processing blocksize (default=%(default)s). For bizarre reasons, pylidar "+
            "uses the square of this as the number of pulses to work with on each block"))
    p.add_argument("--xmin", type=float,
        help="X coord of western edge of DEM image. Default is X_MIN of input file. ")
    p.add_argument("--xmax", type=float,
        help="X coord of eastern edge of DEM image. Default is X_MAX of input file. ")
    p.add_argument("--ymin", type=float,
        help="Y coord of southern edge of DEM image. Default is Y_MIN of input file. ")
    p.add_argument("--ymax", type=float,
        help="Y coord of northern edge of DEM image. Default is Y_MAX of input file. ")
    p.add_argument("--adjustminmax", action='store_true',
        help="Adjust the min max co-ords based on the real world co-ordinates i.e. xmin = UTM co-ord + 50")    
    p.add_argument("--saveminz", 
        help="Img file in which to save minz image (default will not save this)")
    p.add_argument("--savefilteredminz", 
        help="Img file in which to save filtered minz image (default will not save this)")
    p.add_argument("--heightgain", type=float, default=100.0, 
        help="Gain to use for scaling height field (default=%(default)s)")
    p.add_argument("--heightoffset", type=float, default=-500.0, 
        help="Offset to use for scaling height field (default=%(default)s)")
    p.add_argument("--firstfilter", default=None)
    p.add_argument("--pmffilter", action='store_true')
    p.add_argument("--createlas", action='store_true')
    p.add_argument("--interp", default='pynn',
        help="Interpolation method (default is 'pynn' natural neightbour in pylidar - other option matplotlib 'mann' nat neigh)")
    p.add_argument("--saveinterp", 
        help="Img file in which to save interpolated image (default will not save this)")            
        
    cmdargs = p.parse_args()
    
    return cmdargs


def main():
    """
    Main routine
    """
    cmdargs = getCmdargs()
    info = getHeaderInfo(cmdargs.infile)
    userMetadata = json.loads(info['USER_META_DATA'])
    rotationMatrix = numpy.array(userMetadata['Transform'])
    ctrLocal = numpy.array([0, 0, 0, 1])
    (ctrWorldx,  ctrWorldy,  ctrWorldz, constant) = numpy.dot(rotationMatrix, ctrLocal)
       
    defaultBounds(cmdargs, info)
    
    if cmdargs.adjustminmax:
        cmdargs.xmin = numpy.floor((ctrWorldx + cmdargs.xmin)/cmdargs.resolution)*cmdargs.resolution
        cmdargs.xmax = numpy.floor((ctrWorldx + cmdargs.xmax)/cmdargs.resolution)*cmdargs.resolution
        cmdargs.ymin = numpy.floor((ctrWorldy + cmdargs.ymin)/cmdargs.resolution)*cmdargs.resolution
        cmdargs.ymax = numpy.floor((ctrWorldy + cmdargs.ymax)/cmdargs.resolution)*cmdargs.resolution
    
    print("Adjusted bounds:", cmdargs.xmin, cmdargs.xmax, cmdargs.ymin, cmdargs.ymax)
    
    zNull = 0
    
    print("Creating minZ")
    minZ = createInitialMinzImg(cmdargs, zNull)
    if cmdargs.saveminz is not None:
        saveImage(cmdargs.saveminz, minZ, cmdargs.xmin, 
            cmdargs.ymax, cmdargs.resolution, info['SPATIAL_REFERENCE'], zNull)
               

    if cmdargs.firstfilter == "slope":
        print("Applying Slope filter")
        filtered = calcSlope(minZ[0], cmdargs.resolution, zNull)
        print('filtered', filtered.mean())
    elif cmdargs.firstfilter == "median":       
        print("Applying Median filter")
        filtered = medianFilter(minZ[0], zNull)
    
    if cmdargs.pmffilter:             
        print("Applying PMF")  
        goodDataMask = (filtered != zNull)
        filteredOut = pmfjaz.applyPMF(filtered, goodDataMask, cmdargs.resolution, zNull, initWinSize=1, maxWinSize=12, winSizeInc=1, slope=0.3, dh0=0.3, dhmax=5, expWinSizes=False)

        if cmdargs.savefilteredminz is not None:
            saveImage(cmdargs.savefilteredminz, filteredOut, cmdargs.xmin,  
                cmdargs.ymax, cmdargs.resolution, info['SPATIAL_REFERENCE'], zNull)
    else:
        if cmdargs.firstfilter != None:
            filteredOut = filtered
        else:
            filteredOut = minZ[0]                
    
    print("Updating columns")       
    updateClassAndHeight(cmdargs, filteredOut, zNull)   
    
    if cmdargs.createlas:
        outfile = "%s_%s.las" %(cmdargs.infile.split('.')[0], (float(cmdargs.resolution)*100))
        if os.path.exists(outfile):
            os.remove(outfile)
        cmd = "pylidar_translate -i %s -o %s -f LAS" %(cmdargs.infile, outfile)
        os.system(cmd)
    
    ##do interpolation
    filteredOutMask = (filteredOut==zNull)
    
    if cmdargs.saveminz is not None:
        print("Doing interpolation")
        #read in the min zyx image and use filtered mask - lets any filter combo be used before interp function run
        inImage = gdal.Open(cmdargs.saveminz)
        nCols = inImage.RasterXSize
        nRows = inImage.RasterYSize
        geotransform = inImage.GetGeoTransform()
        TLX = geotransform[0]
        TLY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        
        proj = inImage.GetProjection()
        nulVal = inImage.GetRasterBand(1).GetNoDataValue()
        print(nCols, nRows)

      
        z = ((inImage.GetRasterBand(1)).ReadAsArray())
        x = ((inImage.GetRasterBand(2)).ReadAsArray())
        y = ((inImage.GetRasterBand(3)).ReadAsArray())

        mz = numpy.ma.array(z, mask=filteredOutMask).compressed()
        mx = numpy.ma.array(x, mask=filteredOutMask).compressed()
        my = numpy.ma.array(y, mask=filteredOutMask).compressed()
        
        # Generate a regular grid to interpolate the data.
        xi = numpy.linspace(cmdargs.xmin, cmdargs.xmax-cmdargs.resolution, nCols)
        yi = numpy.linspace(cmdargs.ymin, cmdargs.ymax-cmdargs.resolution, nRows)
        xi, yi = numpy.meshgrid(xi, yi)
        print(xi[0:10])
        print(xi.shape)
        print(yi.shape)
        
#        ## Interpolate the values of z for all points in the rectangular grid
#        ## Method 1 - Interpolate by matplotlib delaunay triangularizatio and nearest neigh. PLEASE NOTE! THIS FAILS QUITE OFTEN (http://matplotlib.org/api/mlab_api.html#matplotlib.mlab.griddata) But there might be a solution - install mpl_toolkits.natgrid (http://matplotlib.org/mpl_toolkits/)
#        #zi = ml.griddata(mx,my,mz,xi,yi,interp='nn') #interpolation is 'nn' by default (natural neighbour based on delaunay triangulation) but 'linear' is faster (see http://matplotlib.1069221.n5.nabble.com/speeding-up-griddata-td20906.html)
        #surfaceImg = numpy.flipud(zi)
        
        #use the pylidar natural neighbour interpolation routine
        
        if cmdargs.interp == 'pynn':
            surfaceImg = interpolation.interpGrid(mx, my, mz, (xi,yi), 'pynn')
            surfaceImg = numpy.flipud(surfaceImg)
        elif cmdargs.interp == 'mann':
            zi = ml.griddata(mx,my,mz,xi,yi,interp='nn') #interpolation is 'nn' by default (natural neighbour based on delaunay triangulation) but 'linear' is faster (see http://matplotlib.1069221.n5.nabble.com/speeding-up-griddata-td20906.html)
            surfaceImg = numpy.flipud(zi)     
     
        if cmdargs.saveinterp is not None: 
            saveImage(cmdargs.saveinterp, surfaceImg, cmdargs.xmin,  
                    cmdargs.ymax, cmdargs.resolution, proj, zNull)


def createInitialMinzImg(cmdargs, zNull):
    """
    Create an image array of the minimum Z value for each pixel. Return a 
    2-d array of minZ values for each pixel. 
    """
    dataFiles = lidarprocessor.DataFiles()
    
    # This should be using READ, but because of a bug somewhere, needs UPDATE, so that
    # the call in updateClassAndHeight() will also work. 
    # See https://bitbucket.org/chchrsc/pylidar/issues/8/cannot-open-same-spdv4-file-twice-in-one
    dataFiles.input1 = lidarprocessor.LidarFile(cmdargs.infile, lidarprocessor.UPDATE)  

    otherargs = lidarprocessor.OtherArgs()

    controls = lidarprocessor.Controls()
    controls.setWindowSize(cmdargs.blocksize)
    controls.setSpatialProcessing(False)
    
    # Set up the minZ image array, along with appropriate coordinates relating it to 
    # the real world (i.e. xMin, yMax, res). 
    otherargs.res = cmdargs.resolution
    otherargs.zNull = zNull
    otherargs.xMin = cmdargs.xmin
    otherargs.yMax = cmdargs.ymax
    nCols = int(numpy.ceil((cmdargs.xmax - cmdargs.xmin) / otherargs.res))
    nRows = int(numpy.ceil((cmdargs.ymax - cmdargs.ymin) / otherargs.res))
    otherargs.minZ = numpy.zeros((3, nRows, nCols), dtype=numpy.float32)
    otherargs.minZ.fill(otherargs.zNull)

    lidarprocessor.doProcessing(doMinZ, dataFiles, otherArgs=otherargs, controls=controls)
    
    
    
    return otherargs.minZ                                   
    

def doMinZ(data, otherargs):
    x = data.input1.getPoints(colNames='X')
    y = data.input1.getPoints(colNames='Y')
    z = data.input1.getPoints(colNames='Z')
    
    updateMinZ(x, y, z, otherargs.minZ, otherargs.xMin, 
        otherargs.yMax, otherargs.res, otherargs.zNull)


@jit
def updateMinZ(x, y, z, minzImg, xMin, yMax, res, zNull):
    """
    Called from doProcessing(). 
    
    For the current block of points, given by the x, y, z coord arrays, update the minZ
    image array. Note that the minZ array starts full of nulls, and each time this function 
    is called (for the next block of points), we update whichever pixels are relevant. 
    """
    numPts = x.shape[0]
    (nRows, nCols) = (minzImg.shape[1],minzImg.shape[2])
    #print(yMax,y.min(),y.max())
    
    # Loop explicitly over all points. 
    for i in range(numPts):
        # Calculate the row/column in the image, for the (x, y) coords of the current point
        r = int((yMax - y[i]) / res)
        c = int((x[i] - xMin) / res)
        
        if r >= 0 and r < nRows and c >= 0 and c < nCols:
            # If this is the first point in this pixel, or if this new point has lower Z than
            # previous, then replace the minZ value
            if minzImg[0, r, c] == zNull or minzImg[0, r, c] > z[i]:
                minzImg[0, r, c] = z[i]
                minzImg[1, r, c] = x[i]
                minzImg[2, r, c] = y[i]


def updateClassAndHeight(cmdargs, zArrFiltered, zNull):
    """
    Run through the input SPD file, updating the CLASSIFICATION field and the HEIGHT
    field, based on the minZ array and the PMF-filtered copy of it. 

    """
    dataFiles = lidarprocessor.DataFiles()
    
    dataFiles.input1 = lidarprocessor.LidarFile(cmdargs.infile, lidarprocessor.UPDATE)  

    otherargs = lidarprocessor.OtherArgs()

    controls = lidarprocessor.Controls()
    controls.setWindowSize(cmdargs.blocksize)
    controls.setSpatialProcessing(False)
    
    otherargs.zArrFiltered = zArrFiltered
    otherargs.zThresh = 0.5    # metres. 
    otherargs.xMin = cmdargs.xmin
    otherargs.yMax = cmdargs.ymax
    otherargs.resolution = cmdargs.resolution
    otherargs.heightGain = cmdargs.heightgain
    otherargs.heightOffset = cmdargs.heightoffset
    otherargs.zNull = zNull

    lidarprocessor.doProcessing(doUpdate, dataFiles, otherArgs=otherargs, controls=controls)


def doUpdate(data, otherargs):
    """
    Called from doProcessing. 
    
    Update the CLASSIFICATION and HEIGHT fields, based on how far the points are from the
    PMF-filtered ground image. If a point was classified as ground on input, and is within
    a threshold distance of the filtered ground surface, then it is still ground, otherwise
    it is set to unclassified. For all points, a height value is calculated, above the pixel 
    height. 
    
    """
    x = data.input1.getPoints(colNames='X')
    y = data.input1.getPoints(colNames='Y')
    z = data.input1.getPoints(colNames='Z')
    classification = data.input1.getPoints(colNames='CLASSIFICATION')
    height = numpy.zeros(len(classification), dtype=numpy.float32)
    calcNewClassAndHeight(x, y, z, otherargs.resolution, classification, height, 
        otherargs.zArrFiltered, otherargs.zThresh, otherargs.xMin, otherargs.yMax, otherargs.zNull)

    data.input1.setPoints(classification, colName='CLASSIFICATION')
    data.input1.setScaling('HEIGHT', lidarprocessor.ARRAY_TYPE_POINTS, otherargs.heightGain, 
        otherargs.heightOffset)
    data.input1.setPoints(height, colName='HEIGHT')


@jit
def calcNewClassAndHeight(x, y, z, res, classification, height, zArrFiltered, zThresh, xMin, yMax, zNull):
    """
    Calculate the new point classification, and the height above pixel height. 
    """
    numPts = x.shape[0]
    (nRows, nCols) = zArrFiltered.shape

    # Loop explicitly over all points. 
    for i in range(numPts):
        # Calculate the row/column in the image, for the (x, y) coords of the current point
        r = int((yMax - y[i]) / res)
        c = int((x[i] - xMin) / res)
        
        if r >= 0 and r < nRows and c >= 0 and c < nCols:
            height[i] = z[i] - zArrFiltered[r, c]
            zDiff = numpy.absolute(height[i])
            if zDiff <= 0.001:
                classification[i] = generic.CLASSIFICATION_GROUND
            else:     
                classification[i] = 1

def saveImage(outfile, imgArr, xmin, ymax, res, proj, nullVal):
    """
    Save the given image array as a GDAL raster file. Only saves a single band. 
    """
    
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
    print("proj", proj)      
    if proj != None and len(proj)>50:
        ds.SetProjection(proj)
    ds.SetGeoTransform((xmin, res, 0, ymax, 0, -res))


def defaultBounds(cmdargs, info):
    """
    Fill in default bounds for image
    """
    # Default values for xmin and ymax
    if cmdargs.xmin is None or cmdargs.xmax is None or cmdargs.ymin is None or cmdargs.ymax is None:
        if cmdargs.xmin is None:
            cmdargs.xmin = info['X_MIN']
        if cmdargs.xmax is None:
            cmdargs.xmax = info['X_MAX']
        if cmdargs.ymin is None:
            cmdargs.ymin = info['Y_MIN']
        if cmdargs.ymax is None:
            cmdargs.ymax = info['Y_MAX']


def getHeaderInfo(infile):
    """
    Get Lidar header info. This is a hack. I really should be using
        generic.getLidarFileInfo(infile).header
    However, this opens the file with READ mode, which then makes 
    subsequent opening with UPDATE impossible. So, I kludge here, opening with UPDATE,
    too. This works directly for only SPD V4 files, not other files. Sigh.....
    
    """
    import h5py
    fileHandle = h5py.File(infile, 'r+')
    headerDict = {}
    headerDict.update(fileHandle.attrs)
        
    return headerDict
    
    
def calcSlope(dem, pixSize, zNull):
    """
    Calculate slope and aspect from the dem. Uses Fleming and Hoffer's algorithm. 
    Slope is returned as percent, i.e. 100*rise/run
    Aspect is returned as degrees clockwise from "north" (assuming that the grid 
    is aligned with north), with only positive values (i.e. 0 to 360). 
    
    """
    from scipy.ndimage import convolve
    
    # Kernels for differentiation in x and y directions
    diffX = numpy.array([[0.5,0,-0.5]])
    diffY = numpy.array([[-0.5],[0],[0.5]])

    # Partial derivatives in x and y directions.     
    dzdx = convolve(dem.astype(numpy.float32), diffX) / float(pixSize)
    dzdy = convolve(dem.astype(numpy.float32), diffY) / float(pixSize)
    
    # Trim off the rubbish at the edges. 
    w = 1   # Margin width - 1 pixel
    #dzdx = dzdx[w:-w,w:-w]
    #dzdy = dzdy[w:-w,w:-w]
    
    pcntSlope = abs(100.0 * numpy.sqrt(dzdx**2 + dzdy**2))
    pcntSlope = pcntSlope.astype(numpy.int16)
    
    validSlope = numpy.where(numpy.logical_and(pcntSlope>0.0,pcntSlope<=1000.0), dem, zNull)
    
    return validSlope    
    
def medianFilter(zArr, zNull):    
    zArrMedian = median_filter(zArr, size=3) # Median window size of 3x3, but perhaps 5x5 would be better? Not sure.....
    medianDiff = (zArr - zArrMedian)
    # Knock out pixels which are too high above the median
    diffThresh = 0.2   # Just a guess...... this means 2 metres above the surrounding pixels
    tooHigh = (medianDiff > diffThresh)
    zArr[tooHigh] = zNull
    
    return zArr
    
#def createInterpolation(cmdargs, filteredOut, zNull):
#    """
#    Create an image array for the interpolated Z value using the filtered Z array
#    as input. Return a  2-d array of minZ values for each pixel.
#    """
#    dataFiles = lidarprocessor.DataFiles()
#    
#    # This should be using READ, but because of a bug somewhere, needs UPDATE, so that
#    # the call in updateClassAndHeight() will also work. 
#    # See https://bitbucket.org/chchrsc/pylidar/issues/8/cannot-open-same-spdv4-file-twice-in-one
#    dataFiles.input1 = lidarprocessor.LidarFile(cmdargs.infile, lidarprocessor.UPDATE)  
#
#    otherargs = lidarprocessor.OtherArgs()
#
#    controls = lidarprocessor.Controls()
#    controls.setWindowSize(cmdargs.blocksize)
#    controls.setSpatialProcessing(False)
#    controls.setOverlap(20)
#    
#    # Set up the image array for interpolation, along with appropriate coordinates relating it to 
#    # the real world (i.e. xMin, yMax, res). 
#    otherargs.res = cmdargs.resolution
#    otherargs.zNull = zNull
#    otherargs.xMin = cmdargs.xmin
#    otherargs.xMax = cmdargs.xmin
#    otherargs.yMin = cmdargs.ymin
#    otherargs.yMax = cmdargs.ymax
#    otherargs.interp = cmdargs.interp
#    otherargs.nCols = int(numpy.ceil((cmdargs.xmax - cmdargs.xmin) / otherargs.res))
#    otherargs.nRows = int(numpy.ceil((cmdargs.ymax - cmdargs.ymin) / otherargs.res))
#    otherargs.surface = numpy.zeros((otherargs.nRows, otherargs.nCols, 3), dtype=numpy.float32)
#    otherargs.surface.fill(otherargs.zNull)
#    lidarprocessor.doProcessing(doInterpolation, dataFiles, otherArgs=otherargs, controls=controls)
#    #otherargs.surface = runInterp(otherargs.surface, otherargs.interp, otherargs.zNull, otherargs.xMin, otherargs.xMax, otherargs.yMin, otherargs.yMax, otherargs.nRows, otherargs.nCols)    
#    #print("interp", otherargs.surface.mean()) 
#    
#    return otherargs.surface
#    
#    
#def doInterpolation(data, otherargs):
#    classification = data.input1.getPoints(colNames='CLASSIFICATION')
#    x = data.input1.getPoints(colNames='X')
#    y = data.input1.getPoints(colNames='Y')
#    z = data.input1.getPoints(colNames='Z')
#    #print("doInterp",z.mean())
#    #otherargs.surface = numpy.vstack((x,y,z)) 
#    
#    
#    otherargs.surface = updateSurface(x, y, z, classification, otherargs.surface, otherargs.xMin, otherargs.xMax, otherargs.yMin, 
#        otherargs.yMax, otherargs.res, otherargs.interp, otherargs.zNull)
#        
#    #print("surface", (otherargs.surface))
#    #otherargs.surface = runInterp(x, y, z, otherargs.surface, otherargs.interp, otherargs.zNull, otherargs.xMin, otherargs.xMax, otherargs.yMin, otherargs.yMax, otherargs.nRows, otherargs.nCols)    
#    #print("interp", otherargs.surface.mean()) 
#    
#       
#@jit
#def updateSurface(x, y, z, classification, surfaceImg, xMin, xMax, yMin, yMax, res, interp, zNull):
#    """
#    Called from doProcessing(). 
#    
#    For the current block of points, given by the x, y, z coord arrays, do interpolation of the Z
#    image array where classification is ground. Note that the surface array starts full of nulls, and 
#    each time this function is called (for the next block of points), we update whichever pixels are relevant. 
#    """
#    numPts = x.shape[0]
#    #print(numPts)
#    (nRows, nCols) = (surfaceImg.shape[0], surfaceImg.shape[1])
#    #print(classification.min(),classification.max())
#    #print("z", z[classification==2].mean())
#    
#    # Loop explicitly over all points. 
#    for i in range(numPts):
#        # Calculate the row/column in the image, for the (x, y) coords of the current point
#        r = int((yMax - y[i]) / res)
#        c = int((x[i] - xMin) / res)
#        #if classification[i] == 2:
#            #print(z[i])
#
#        if r >= 0 and r < nRows and c >= 0 and c < nCols and classification[i]==2:            
#            if surfaceImg[r, c, 2] == zNull or surfaceImg[r, c, 2] > z[i]:               
#                surfaceImg[r, c, 0] = x[i]
#                surfaceImg[r, c, 1] = y[i]
#                surfaceImg[r, c, 2] = z[i]
#    print(surfaceImg)                      
#    
#     
##def runInterp(surfaceImg, interp, zNull, xMin, xMax, yMin, yMax, nRows, nCols):
##    """
##    Do the interpolation using the specified method
##    """
##    # Generate a regular grid to interpolate the data.
##    xi = numpy.linspace(xMin, xMax, nCols)
##    yi = numpy.linspace(yMin, yMax, nRows)
##    xi, yi = numpy.meshgrid(xi, yi)
##    nullMask = (surfaceImg[2]==zNull)
##    xm = ma.array(surfaceImg[0], mask = nullMask).compressed()
##    ym = ma.array(surfaceImg[1], mask = nullMask).compressed()
##    zm = ma.array(surfaceImg[2], mask = nullMask).compressed()
##    zi = ml.griddata(xm,ym,zm,xi,yi,interp='nn') #interpolation is 'nn' by default (natural neighbour based on delaunay triangulation) but 'linear' is faster (see http://matplotlib.1069221.n5.nabble.com/speeding-up-griddata-td20906.html)
##    ziflip = np.flipud(zi)
##    print('zi',ziflip.max())
##      
##    
##    try:
##        #use try except to allow function to still work when interpolator errors are raised i.e. less 4 points and var. X and Y less 4.
##        print(pxlCoords[0])
##        
##               
##        surfaceImg = interpolation.interpGrid(xm, ym, zm, pxlCoords, interp)
##        #print("here", surfaceImg.max())
##        surfaceImg = numpy.expand_dims(surfaceImg, axis=0).astype(numpy.float32)
##        overshootMask = numpy.isnan(surfaceImg)
##        surfaceImg[overshootMask] = zNull
##        #print(out)
##    except Exception:
##        surfaceImg.fill(zNull)
##        #print('except')
##    else:
##        surfaceImg.fill(zNull)    
##    return(surfaceImg)

if __name__ == "__main__":
    main()
