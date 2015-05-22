#! /usr/bin/env python
from __future__ import division, print_function
import scipy as sp
from numpy import atleast_2d
from sklearn.gaussian_process import GaussianProcess
import random
from matplotlib import pyplot as plt
import time
import scipy.spatial as spsp
import theano
import theano.tensor as the


def remove_duplicate_vectors(X):
    X_purged = []
    for v in X:
        if v not in sp.asarray(X_purged):
            X_purged.append(v)
    return sp.asarray(X_purged)


def test_points(X, yprime, reach, n_test_pts):
    yp = sp.absolute(yprime)
    reaches = reach * sp.exp(- yp / yp.mean())
    # reaches[:,:] = re.
    X_test = []
    for point, r in zip(X, reaches):
        # points within rectangle. meshgrid will work in > 2D for numpy > 1.7
        xs = sp.asarray([sp.meshgrid(sp.linspace(point[i] - r[i], point[i] + r[i], n_test_pts)) for i in range(n_features)])
        X_test.append(sp.vstack([x.ravel() for x in xs]).reshape(n_features,-1).T)
    X_test = sp.asarray(X_test).reshape(-1,n_features)
    X_test = remove_duplicate_vectors(X_test)    
    X_test = sp.sort(X_test, axis = 0)
    return X_test


def is_point_within_rectangle(point, Xmin, Xmax):
    return sp.alltrue(point >= Xmin) and sp.alltrue(point <= Xmax)


def twoD_gauss(x, y, x0, sigma, angle):
    R = sp.array([[the.cos(angle), the.sin(angle)],[-the.sin(angle), the.cos(angle)]])
    Rx = R[0,0] * x + R[0,1] * y
    Ry = R[1,0] * x + R[1,1] * y
    return the.exp(- 0.5 * ((Rx - x0[0]) / sigma[0])**2 - 0.5 * ((Ry - x0[1]) / sigma[1])**2)


def target_fun(X):
    X = sp.asarray(X)
    if len(sp.shape(X)) == 0:
        thX = the.scalar('thX')
    elif len(X.shape) == 1:
        thX = the.dvector('thX')
        y = 0.8 * twoD_gauss(thX[0], thX[1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4) + 1.2 * twoD_gauss(thX[0], thX[1], sp.array([3,0]), sp.array([1,1]), 0)
        fun = theano.function([thX], y)
        return fun(X)
    elif len(X.shape) == 2:
        thX = the.dmatrix('thX')
        y = 0.8 * twoD_gauss(thX[:,0], thX[:,1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4) + 1.2 * twoD_gauss(thX[:,0], thX[:,1], sp.array([3,0]), sp.array([1,1]), 0)
        fun = theano.function([thX], y)
        return fun(X)
    else:
        print("Bad Input")
    

def dtarget_fun(X):
    results = []
    X = sp.asarray(X)
    thX = the.dvector('thX')
    y = 0.8 * twoD_gauss(thX[0], thX[1], sp.array([-1,-1]), sp.array([3,2]), sp.pi/4) + 1.2 * twoD_gauss(thX[0], thX[1], sp.array([3,0]), sp.array([1,1]), 0)
    grady = the.grad(y, thX)
    dfun = theano.function([thX], grady)
    
    if len(sp.shape(X)) == 0:
        thX = the.scalar('thX')
        return dfun(X)
    elif len(X.shape) == 1:
        return dfun(X)
    elif len(X.shape) == 2:
        results = []
        for x in X:
            results.append(dfun(x))
        return sp.array(results)
    else:
        print("Bad Input")
    

def select_next_x(sort_order, x_list, old_x):
    x_list = x_list[sort_order]
    for x in x_list:
        if x not in old_x:
            return x
    

def dynamics_step(pos, vel, deltat, posminmax):
    """
    Basic: x_1 = x_0 + v_0 * dt + 0.5 * a * (dt)**2
    Bounce back from extremes
    """
    dpos = vel * deltat - 0.5 * deltat**2 * dtarget_fun(pos)
    dvel = - deltat * dtarget_fun(pos)
    if (pos + dpos <= posminmax[1]) and (pos + dpos >= posminmax[0]):
        return pos + dpos, vel + dvel
    else:
        return pos - dpos, - (vel + dvel)


def highest_MSE_x(gp, X_test, X_old, Xminmax):
    """
    Given a trained Gaussian Process gp, predict the values y_pred on a given grid x_test.
    Return the x value corresponding to the highest predicted variance which is not present in x_old
    and is within the given extrema xminmax.
    """
    
    y_pred, MSE = gp.predict(atleast_2d(X_test), eval_MSE=True)
    max_pred_err = (sp.sqrt(MSE)).max()
    sort_order = sp.argsort(MSE)[::-1]
    X_t = X_test[sort_order]
    X_t = list(X_t)
    while len(X_t) > 1:
        Xnew = X_t[0]
        if is_point_within_rectangle(Xnew, Xminmax[0], Xminmax[1]):
            return Xnew, y_pred, MSE, max_pred_err
        else:
            X_t.pop(0)

            
def draw_2Dplot(ax, X, y, X_test, y_test, MSE, Xnew):
    ax0 = ax[0,0]
    bubbles = sp.exp(-MSE)
    bubbles = sp.array((bubbles - bubbles.min()) / bubbles.std() * 40 + 1, dtype='int')
    ax0.clear()
    ax0.scatter(X[:,0], X[:,1], s = 100, marker = 's', c = y, cmap = 'YlGnBu', alpha = 0.8)
    ax0.scatter(X_test[:,0], X_test[:,1], s = bubbles, c = y_test, cmap = 'YlGnBu', alpha = 0.4, edgecolors='none')
    ax0.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)
    ax0.set_xlabel('$CV 1$')
    ax0.set_ylabel('$CV 2$')
    ax0.set_xlim(Xmin[0], Xmax[0])
    ax0.set_ylim(Xmin[1], Xmax[1])

    
def draw_2Dsecondaryinfo(ax, gp, Xnew, Xminmax, step):
    Xmin, Xmax = Xminmax
    ax1 = ax[0,1]
    ax2 = ax[1,0]
    ax3 = ax[1,1]

    X_grid = sp.meshgrid(sp.linspace(Xmin[0], Xmax[0], 50), sp.linspace(Xmin[1], Xmax[1], 50))
    X_grid = sp.vstack([x.ravel() for x in X_grid]).reshape(n_features,-1).T

    y_grid, MSE_grid = gp.predict(X_grid, eval_MSE=True)
    ax2.clear()
    ax2.scatter(X_grid[:,0], X_grid[:,1], marker = 'h', s = 200, c = MSE_grid, cmap = 'YlGnBu', alpha = 1, edgecolors='none')
    ax2.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)


    ax3.clear()
    ax3.scatter(X_grid[:,0], X_grid[:,1], s = 200, c = y_grid, cmap = 'YlGnBu', alpha = 1, edgecolors='none')
    ax3.scatter(Xnew[0], Xnew[1], marker='h', s = 400, c = 'g', alpha = 0.5)

    if step < 1:
        y_target = target_fun(X_grid)
        ax1.scatter(X_grid[:,0], X_grid[:,1], s = 200, marker='s', c = y_target, cmap = 'YlGnBu', alpha = 1, edgecolors='none')
    for axx in [ax1, ax2, ax3]:
        axx.set_xlabel('$CV 1$')
        axx.set_ylabel('$CV 2$')
        axx.set_xlim(Xmin[0], Xmax[0])
        axx.set_ylim(Xmin[1], Xmax[1])
    

#######################################################
#######################################################
#######################################################

Xmin, Xmax = sp.array([-6., -6.]), sp.array([6., 6.])
theta = 1.0e-0
nugget = 1.0e-8
method = 'local_high_variance'# 'highest_variance_grid', 'local_high_variance'
convergence_threshold = 1.0e-3
n_features = 2

# Setup Gaussian Process
# gp = GaussianProcess(corr='linear', theta0=1e-1, thetaL=1e-4, thetaU=1e+0, normalize=True, nugget=nugget)
gp = GaussianProcess(corr='squared_exponential', theta0=theta, thetaL=1e-2, thetaU=1e+1, nugget = nugget, normalize=False)

# first evaluation points drawn at random from range
X = sp.array([ [random.uniform(Xmin[0], Xmax[0]), random.uniform(Xmin[1], Xmax[1])] for i in range(2)])
y = target_fun(X)
yprime = dtarget_fun(X)

# teach the first 2 trial points
gp.fit(atleast_2d(X),y)

# setup plot
plt.close('all')
plt.ion()
fig, ax = plt.subplots(nrows=2, ncols=2)
fig, ax = plt.subplots(2,2,figsize=(15,15))

step = 0

if method == 'local_high_variance':
    """
    --- Go wherever the predicted variance is highest around data points ---
    few points drawn at random -> evaluate energy + force
    train GP
    pick new x from neighbourhood of previous ones as the one with largest MSE
    evaluate energy landscape via GP for next value of x
    """
    n_test_pts = 20
    reach = 2

    X_test = test_points(X, yprime, reach, n_test_pts)

    Xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, X_test, X, [Xmin, Xmax])
    
    draw_2Dplot(ax, X, y, X_test, y_pred, MSE, Xnew)
    draw_2Dsecondaryinfo(ax, gp, Xnew, [Xmin, Xmax], step)
    fig.canvas.draw()

    while max_pred_err > convergence_threshold:
        # do new calculation on the point with largest predicted variance
        X = sp.vstack((X, Xnew))
        y = target_fun(X).ravel() # wasteful, I'm recalculating all y. But who cares for now
        yprime = dtarget_fun(X) # wasteful, I'm recalculating all y. But who cares for now
        
        gp.fit(atleast_2d(X), y)

        X_test = test_points(X, yprime, reach, n_test_pts)

        Xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, X_test, X, [Xmin, Xmax])
        
        draw_2Dplot(ax, X, y, X_test, y_pred, MSE, Xnew)
        draw_2Dsecondaryinfo(ax, gp, Xnew, [Xmin, Xmax], step)
        fig.canvas.draw()
        # print("mean force: %.03f" % sp.absolute(yprime).mean())
        
        print("Step %03d | MAX ERROR: %.03f" % (step, max_pred_err))
        # plt.savefig('ajvar-%03d.png' % step)
        # time.sleep(3)
        step += 1

    
elif method == 'highest_variance_grid':
    """
    --- Go wherever the predicted variance is highest ---
    2 points drawn at random -> evaluate energy + force
    train GP
    evaluate energy landscape via GP for a given range of values 
    err
    """
    n_test_pts = 200
    x_test = sp.linspace(xmin, xmax, n_test_pts)

    y_pred, MSE = gp.predict(atleast_2d(x_test).T, eval_MSE=True)
    max_pred_err = (sp.sqrt(MSE)).max()
    sort_order = sp.argsort(MSE)[::-1]
    lines = draw_plot(fig, x, y, x_test, y_pred, MSE)

    xnew = select_next_x(sort_order, x_test, x)
    while max_pred_err > convergence_threshold:
        # do new calculation on the point with largest predicted variance
        x = sp.hstack((x, xnew))
        y = target_fun(x).ravel() # wasteful, I'm recalculating all y. But who cares for now

        gp.fit(atleast_2d(x).T,y)

        xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, x_test, x, [xmin, xmax])
        
        # y_pred, MSE = gp.predict(atleast_2d(x_test).T, eval_MSE=True)
        # max_pred_err = (sp.sqrt(MSE)).max()
        # sort_order = sp.argsort(MSE)[::-1]
        # xnew = select_next_x(sort_order, x_test, x)
        lines = draw_plot(fig, x, y, x_test, y_pred, MSE)
        print("Step %03d | MAX ERROR: %.03f" % (step, max_pred_err))
        # pl.savefig('ajvar-%03d.png' % step)
        # time.sleep(.4)
        step += 1

