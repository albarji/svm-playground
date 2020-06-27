"""
Graphical User Interface to visualize toy SVM models.

Author: Álvaro Barbero Jiménez <albarjip@gmail.com>
"""
import numpy as np
import holoviews as hv
from holoviews import opts
from holoviews import streams
import matplotlib
from functools import partial

from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR, OneClassSVM

hv.extension('bokeh', 'matplotlib')

# Plot limits
XMIN = YMIN = 0
XMAX = YMAX = 1

### Basic plots

def decision_surface_plot(model=None):
    """Returns a Holoviews object plotting the decision surface for a classification model"""
    delta = 0.01
    x = np.arange(XMIN, XMAX + delta, delta)
    y = np.arange(YMIN, YMAX + delta, delta)
    xs, ys = np.meshgrid(x, y)
    if model is not None:
        zs = model.decision_function([[x, y] for x, y in zip(xs.flatten(), ys.flatten())])
        zs = zs.reshape(xs.shape)
    else:
        zs = np.zeros(xs.shape)
    img = hv.Image((x, y, zs)).opts(colorbar=True, cmap='bkr')
    contour = hv.operation.contours(img, levels=[-1.0, 0.0, 1.0]).options(
        cmap='coolwarm', 
        tools=['hover'], 
        line_width=5,
        show_legend=False
    )
    
    return img * contour

def points_plot(points, pointshover=True):
    """Returns a Holoviews object plotting a set of points with class labels"""
    return hv.Points(points, vdims='class').opts(
        color='class', 
        cmap={1: 'red', -1: 'blue'}, 
        line_color='black',
        line_width=1,
        size=10,
        tools=['hover'] if pointshover else []
    )

def regression_plot(model=None, epsilon=0):
    """Returns a Holoviews object plotting the regression curve of a SVR model"""
    delta = 0.01
    x = np.arange(XMIN, XMAX + delta, delta)
    if model is not None:
        y = model.predict([[d] for d in x])
    else:
        mean = (YMAX+YMIN)/2
        y = np.array([mean for _ in x])

    return (hv.Curve(zip(x, y)).opts(line_width=5)
        * hv.Curve(zip(x, y + epsilon)).opts(color='gray', line_dash='dotted')
        * hv.Curve(zip(x, y - epsilon)).opts(color='gray', line_dash='dotted')
    )

### Specific SVM plots

def update_svmclassification_plot(taps, x, y, x2, y2, kernel, log_C, log_gamma):
    # Record new clicks    
    if None not in [x,y]:
        taps.append((x, y, 1))
    elif None not in [x2, y2]:
        taps.append((x2, y2, -1))
        
    X = [tap[:-1] for tap in taps]
    y = [tap[-1] for tap in taps]
        
    # Update SVM (if data available)
    if len(X) and len(set(y)) > 1:
        if kernel == "Linear":
            model = LinearSVC(C=10**log_C).fit(X, y)
        elif kernel == "Gaussian":
            model = SVC(C=10**log_C, gamma=10**log_gamma).fit(X, y)
    else:
        model = None
    
    # Build plots
    image = decision_surface_plot(model)
    points = points_plot(taps)
    
    return image * points

def svm_classification_plot():
    """Returns a Holoviews DynamicMap with an interactive plot of SVM classification models"""
    taps = []
    return hv.DynamicMap(
        partial(update_svmclassification_plot, taps), 
        streams=[
            streams.SingleTap(transient=True), 
            streams.DoubleTap(rename={'x': 'x2', 'y': 'y2'}, transient=True)        
        ], 
        kdims=['kernel','log_C','log_gamma']
    ).opts(
        width=600,
        height=600,
        title="SVM decision map",
        toolbar=None,
        active_tools=[None]  # TODO: this does nothing. We want to disable panning
    ).redim.range(
        log_C=(-3.0, 6.0),
        log_gamma=(-3.0,3.0),
    ).redim.values(
        kernel=['Linear', 'Gaussian']
    ).redim.default(
        log_C=2,
        log_gamma=1
    )

def update_svmregression_plot(taps, x, y, kernel, log_C, log_gamma, epsilon):
    # Record new clicks    
    if None not in [x,y]:
        taps.append((x, y, 1))
        
    X = [[tap[0]] for tap in taps]
    y = [tap[1] for tap in taps]

    # Update SVM (if data available)
    if len(X):
        if kernel == "Linear":
            model = LinearSVR(C=10**log_C, epsilon=epsilon).fit(X, y)
        elif kernel == "Gaussian":
            model = SVR(C=10**log_C, gamma=10**log_gamma, epsilon=epsilon).fit(X, y)
    else:
        model = None
    
    # Build plots
    curve = regression_plot(model, epsilon)
    points = points_plot(taps, pointshover=False)
    merge = curve * points
    return merge.opts(xlim=(XMIN, XMAX), ylim=(YMIN, YMAX))

def svm_regression_plot():
    """Returns a Holoviews DynamicMap with an interactive plot of SVM regression models"""
    taps = []
    return hv.DynamicMap(
        partial(update_svmregression_plot, taps), 
        streams=[
            streams.SingleTap(transient=True)
        ], 
        kdims=['kernel', 'log_C', 'log_gamma', 'epsilon']
    ).opts(
        width=600,
        height=600,
        title="SVR regression curve",
        toolbar=None,
        active_tools=[None]  # TODO: this does nothing. We want to disable panning
    ).redim.range(
        log_C=(-3.0, 6.0),
        log_gamma=(-3-0,3.0),
        epsilon=(0.01,0.5),
    ).redim.values(
        kernel=['Linear', 'Gaussian']
    ).redim.default(
        log_C=1,
        log_gamma=1,
        epsilon=0.1
    )

def update_svmoneclass_plot(taps, x, y, nu, log_gamma):
    # Record new clicks    
    if None not in [x,y]:
        taps.append((x, y, 1))
        
    X = [tap[:-1] for tap in taps]
        
    # Update SVM (if data available)
    if len(X):
        model = OneClassSVM(nu=nu, gamma=10**log_gamma).fit(X, y)
    else:
        model = None
    
    # Build plots
    image = decision_surface_plot(model)
    points = points_plot(taps, pointshover=False)
    
    return image * points

def svm_oneclass_plot():
    """Returns a Holoviews DynamicMap with an interactive plot of SVM one-class models"""
    taps = []
    return hv.DynamicMap(
        partial(update_svmoneclass_plot, taps), 
        streams=[
            streams.SingleTap(transient=True), 
        ], 
        kdims=['nu', 'log_gamma']
    ).opts(
        width=600,
        height=600,
        title="One-class SVM decision map",
        toolbar=None,
        active_tools=[None]  # TODO: this does nothing. We want to disable panning
    ).redim.range(
        nu=(0.01, 0.99),
        log_gamma=(-3.0,3.0)
    ).redim.default(
        nu=0.5,
        log_gamma=-1
    )