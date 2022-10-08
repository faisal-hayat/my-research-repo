# How K-Means, KNN works

--- ---

## K-Means algorithm working is given as

- Initialize K random centroids.
    1. You could pick K random data points and make those your starting points.
    2. therwise, you pick K random values for each variable.
- For every data point, look at which centroid is nearest to it.
    1. Using some sort of measurement like Euclidean or Cosine distance.
- Assign the data point to the nearest centroid.
- For every centroid, move the centroid to the average of the points assigned to that centroid.
- Repeat the last three steps until the centroid assignment no longer changes.
    1. The algorithm is said to have “converged” once there are no more changes.

--- ---

## How KNN works

- Select the number K of the neighbors
- Calculate the Euclidean distance of K number of neighborsTake the K nearest neighbors as per the calculated Euclidean
  distance.
- Among these k neighbors, count the number of the data points in each category.
- Assign the new data points to that category for which the number of the neighbor is maximum.
- model is ready

--- ---