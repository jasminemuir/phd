#!/usr/bin/env python

import sys
import os
from osgeo import gdal
import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inImage = sys.argv[1]
outImage = sys.argv[2]
numClusters = sys.argv[3]


ds = gdal.Open(inImage)
imageArray = numpy.array(ds.GetRasterBand(int(1)).ReadAsArray())
print("imageArray",imageArray.shape)

imageArray1D = numpy.column_stack([imageArray.flatten()])
print("imageArray1D",imageArray1D.shape)
k_means = KMeans(n_clusters=int(numClusters)+1)
pred = k_means.fit_predict(imageArray1D)


clusterCentres = k_means.cluster_centers_.flatten()

from_values = numpy.argsort(clusterCentres)
to_values = (numpy.arange(from_values.size))
print(from_values,to_values)
sort_idx = numpy.argsort(from_values)
print(sort_idx)

idx = numpy.searchsorted(from_values, pred, sorter = sort_idx)
out = to_values[sort_idx][idx]
print(idx)

labels = k_means.labels_
print(clusterCentres)
print(sort_idx)
print(labels)

classImageArray = out.reshape(imageArray.shape)
print("classImageArray",classImageArray.shape)

plt.imshow(classImageArray)
plt.colorbar()
plt.show()


del ds

