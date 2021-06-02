# Slope_Regression.py: Solve for linear slope and intercept given a set of points
# HP

import numpy as np
import matplotlib.pyplot as plt

def getRegressionSlope(points):
    """
    Takes a given number of points and outputs slope determined by linear regression
    Times can be included, if they are not, the indexes are assumed to be the times, starting from 0
    """
    timeArray = np.ones((len(points), 2))
    timeArray[:,1]  = np.arange(len(points))
    pointArray = np.array(points)
    point_slope, res, _, _ = np.linalg.lstsq(timeArray, np.log(pointArray), rcond=None)
    slope = point_slope[1]
    err = res[0]
    return slope, err

    
def regressionSlopeSeries(points, window):
    """
    Given a series of points and a window size, estimates the slope at each point
    Outputs both the estimated slopes and regression errors for all applicable points
    If times list is provided, each of the included times should match up respectively with the points
    Window should be odd, if it isn't it will automatically be interpreted as one larger than the argument passed
    """
    # cut the window in half so we know how long it extends on both sides
    half_window = window//2
    num_points = len(points)
    slopes = []
    errs = []
    for i in range(num_points):
        # if there are not enough previous values to satisfy window size, continue
        if i < half_window:
            continue
        # if there are not enough values after the current one to satisfy window size, break
        if num_points-i < half_window:
            break
        slope, err = getRegressionSlope(points[i-half_window: i+half_window+1])
        slopes.append(slope)
        errs.append(err)
    slopes = np.array(slopes)
    errs = np.array(errs)
    return slopes, errs

