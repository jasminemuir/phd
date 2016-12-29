#!/usr/bin/env python

import sys
import numpy
from rios import applier
from rios import fileinfo

def doMinimum(info, inputs, outputs, otherargs):
    "Called from RIOS. Average the input files"
    minimum = numpy.zeros(inputs.imgs[0].shape, dtype = numpy.float32)
    for img in inputs.imgs:
        img[numpy.isnan(img)] = otherargs.noDataVal
        
        imgNonNull = (img != otherargs.noDataVal)
        minNull = (minimum == otherargs.noDataVal)
        minimum[minNull] = img[minNull]
        newMin = (imgNonNull & ~minNull & (img < minimum))
        minimum[newMin] = img[newMin]


    outputs.min = minimum.astype(img.dtype)

infiles = applier.FilenameAssociations()
# names of imput images
infiles.imgs = sys.argv[1:]

otherargs = applier.OtherInputs()
otherargs.noDataVal = float(fileinfo.ImageInfo(infiles.imgs[0]).nodataval[0])
print(otherargs.noDataVal)

# Last name given is the output
outfiles = applier.FilenameAssociations()
outfiles.min = "outfile18.img"
controls = applier.ApplierControls()
controls.setFootprintType(applier.UNION)
applier.apply(doMinimum, infiles, outfiles, otherargs, controls=controls)
