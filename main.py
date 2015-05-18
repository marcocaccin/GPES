#! /usr/bin/env python
from __future__ import division, print_function
import scipy as sp
from numpy import atleast_2d
from sklearn.gaussian_process import GaussianProcess
import random
from matplotlib import pyplot as pl
import time
import scipy.spatial as spsp

def target_fun(x):
    x = sp.asarray(x)
    return - 0.15 * x**6 * sp.exp(-sp.absolute(x)) + 0.5*x**2
    

def dtarget_fun(x):
    x = sp.asarray(x)
    return 0.15 * sp.sign(x) * sp.exp(-sp.absolute(x)) * x**6 - 0.15 * 6 * x**5 * sp.exp(-sp.absolute(x)) + x

 
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


def draw_plot(fig, x, y, x_test, y_test, MSE):
    pl.clf()
    maxMSE = sp.argsort(MSE)[-1]
    line1, = pl.plot(x_test, target_fun(x_test), 'r:', label=u'$f(x)$')
    line2, = pl.plot(x, y, 'r.', markersize=10, label=u'Observations')
    line3, = pl.plot(x_test, y_pred, 'b-', label=u'Prediction')
    line4, = pl.fill(sp.concatenate([x_test, x_test[::-1]]),
                     sp.concatenate([y_pred - 1.9600 * sp.sqrt(MSE),
                                     (y_pred + 1.9600 * sp.sqrt(MSE))[::-1]]),
                     alpha=.4, fc='b', ec='None', label='95% confidence interval')
    line5, = pl.plot(x[-1], y[-1], 'go', markersize=20, alpha=0.4)

    pl.xlabel('$x$')
    pl.ylabel('$f(x)$')
    pl.xlim(xmin, xmax)
    pl.ylim(1.5*target_fun(x_test).min(), 1.5*target_fun(x_test).max())

    # pl.legend(loc='upper left')
    pl.draw()
    return [line1, line2, line3, line4]


def highest_MSE_x(gp, x_test, x_old, xminmax):
    """
    Given a trained Gaussian Process gp, predict the values y_pred on a given grid x_test.
    Return the x value corresponding to the highest predicted variance which is not present in x_old
    and is within the given extrema xminmax.
    """
    y_pred, MSE = gp.predict(atleast_2d(x_test).T, eval_MSE=True)
    max_pred_err = (sp.sqrt(MSE)).max()
    sort_order = sp.argsort(MSE)[::-1]
    x_t = x_test[sort_order]

    while x_t.size > 1:
        xnew = x_t[0]
        if (xnew <= xminmax[1]) and (xnew >= xminmax[0]) and (xnew not in x_old):
            return xnew, y_pred, MSE, max_pred_err
        else:
            x_t = list(x_t)
            x_t.pop(0)
            x_t = sp.asarray(x_t)


#######################################################
#######################################################
#######################################################
            
xmin, xmax = -6., 6.

theta = 1.0e-1
nugget = 1.0e-8
method = 'local_high_variance'# 'highest_variance_grid', 'local_high_variance'
convergence_threshold = 1.0e-2


# Setup Gaussian Process
# gp = GaussianProcess(corr='linear', theta0=1e-1, thetaL=1e-4, thetaU=1e+0, normalize=True, nugget=nugget)
gp = GaussianProcess(corr='squared_exponential', theta0=theta, thetaL=1e-2, thetaU=1e+0, nugget = nugget, normalize=False)

# first 2 evaluation points drawn at random from range
x = sp.array([random.uniform(xmin,xmax) for i in range(4)])
y = target_fun(x).ravel()

# teach the first 2 trial points
gp.fit(atleast_2d(x).T,y)

# setup plot
pl.ion()
pl.clf()
fig = pl.figure()

step = 0
if method == 'highest_variance_grid':
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
        time.sleep(.4)
        step += 1

elif method == 'dynamics':
    deltat = .01
    vel = 4.
    n_test_pts = 200
    x_test = sp.linspace(xmin, xmax, n_test_pts)
    y_pred, MSE = gp.predict(atleast_2d(x_test).T, eval_MSE=True)  
    max_pred_err = (sp.sqrt(MSE)).max()
    lines = draw_plot(fig, x, y, x_test, y_pred, MSE)

    xnew, vel = dynamics_step(x[-1], vel, deltat, [xmin, xmax])
    
    while max_pred_err > convergence_threshold:
        # do new "calculation" on the next point in dynamics, unless it is too close to already calculated one
        if sp.absolute(x - xnew).min() > 1.0e-6:
            x = sp.hstack((x, xnew))
        y = target_fun(x).ravel() # wasteful, I'm recalculating all y. But who cares for now

        gp.fit(atleast_2d(x).T,y)
        _, y_pred, _, max_pred_err = highest_MSE_x(gp, x_test, x, [xmin, xmax])
        lines = draw_plot(fig, x, y, x_test, y_pred, MSE)
        xnew, vel = dynamics_step(xnew, vel, deltat, [xmin, xmax])
        print(xnew, vel)
        print("Step %03d | MAX ERROR: %.03f" % (step, max_pred_err))
        # pl.savefig('dynamics-%03d.png' % step)
        step += 1
        #time.sleep(.1)


elif method == 'local_high_variance':
    """
    --- Go wherever the predicted variance is highest around data points ---
    few points drawn at random -> evaluate energy + force
    train GP
    pick new x from neighbourhood of previous ones as the one with largest MSE
    evaluate energy landscape via GP for next value of x  
    """
    n_test_pts = 8
    reach = 1/3 
    x_test = sp.array([sp.linspace(point - reach, point + reach, n_test_pts) for point in x]).flatten()
    x_test = sp.sort(x_test)
    
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

        x_test = sp.array([sp.linspace(point - reach, point + reach, n_test_pts) for point in x]).flatten()
        x_test = sp.sort(x_test)

        xnew, y_pred, MSE, max_pred_err = highest_MSE_x(gp, x_test, x, [xmin, xmax])
        
        lines = draw_plot(fig, x, y, x_test, y_pred, MSE)
        print("Step %03d | MAX ERROR: %.03f" % (step, max_pred_err))
        # pl.savefig('ajvar-%03d.png' % step)
        time.sleep(.3)
        step += 1


    

