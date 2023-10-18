import numpy as np
import pandas as pd
import math


def ApplyFilter(a1, p1, e, n, y):
    timepoints = len(y)
    a, p, F = np.zeros(timepoints+1, dtype = np.double), np.zeros(timepoints+1, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A, P = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)

    a[0], p[0] = a1, p1

    for i in range(0, timepoints):
        F[i] = p[i] + e
        K = p[i]/F[i]
        #predicting
        if np.isnan(y[i]): 
            A[i] = a[i]
            P[i] = p[i]
        else:
            A[i] = a[i] + K*(y[i] - a[i])
            P[i] = K*e
        #predicting
        a[i+1] = A[i]
        p[i+1] = P[i] + n
    
    return a[0:-1], p[0:-1],F,A,P

def ApplySmoothing(a, p, F, y, v, e):
    timepoints=len(y)
    r, N = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)
    A_, P_ = np.zeros(timepoints, dtype = np.double), np.zeros(timepoints, dtype = np.double)

    for i in range(timepoints-1, 0, -1):
        if np.isnan(y[i]): r[i-1], N[i-1] = r[i], N[i]
        else:
            r[i-1] = (v[i]/F[i]) + (e/F[i])*r[i]
            N[i-1] = (1/F[i]) + ((e/F[i])**2)*N[i]
    r_ = (v[0]/F[0]) + (e/F[0])*r[0]
    N_ = (1/F[0]) + ((e/F[0])**2)*N[0]
    
    A_[0] = a[0] + p[0]*r_
    P_[0] = p[0] - (p[0]**2)*N_
    for i in range(1,timepoints):
        A_[i] = a[i] + p[i]*r[i-1]
        P_[i] = p[i] - (p[i]**2)*N[i-1]
    
    return A_, P_, r, N




def LogLikelihood(y, a1, p1, e, n):
    a, p, F, A, P = ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]
    
    L = len(F)*math.log(2*math.pi)
    for i in range(len(F)): L += math.log(F[i]) + v[i]**2/F[i]
    
    return -L/2


def Likelihood_de(y, a1, p1, e, n):
    a, p, F, A, P = ApplyFilter(a1, p1, e, n, y)
    v = [y[i] - a[i] for i in range(len(y))]
    
    dF, dv = [], []
    dF.append(1)
    dv.append(0)
    
    dL = (1/(F[0]**2)*(F[0]-v[0]**2)*dF[0]) + 2*F[0]*v[0]*dv[0]
    for i in range(1, len(F)):
        dF.append((e/F[i-1])**2*dF[i-1]-2*e/F[i-1]+2)
        dv.append(dv[i-1] - ((1-e/(F[i-1])**2)*dv[i-1] + e*v[i-1]*dF[i-1]/(F[i-1]**2)-v[i-1]/F[i-1]))
        dL += 1/(F[i]**2)*((F[i]-v[i]**2)*dF[i] + 2*F[i]*v[i]*dv[i])
    
    return -dL/2


def MaximizeLikelihood(y, a1, p1, e, n):
    L, E, grad = [], [], []
    
    iter = 1
    L.append(LogLikelihood(y, a1, p1, e, n))
    E.append(e)
    grad.append(Likelihood_de(y, a1, p1, e, n))
    print("iter = "+str(iter), "L = "+str(L[-1]), "E = "+str(E[-1]), "Gradient = "+str(grad[-1]))
    
    step = 5
    stop = False
    while stop == False:
        iter += 1
        e = e + step
        L.append(LogLikelihood(y, a1, p1, e, n))
        E.append(e)
        grad.append(Likelihood_de(y, a1, p1, e, n))
        print("iter = "+str(iter), "L = "+str(L[-1]), "E = "+str(E[-1]), "Gradient = "+str(grad[-1]))
        
        if abs(grad[-1])<=1e-7: stop = True
        elif (grad[-1]*grad[-2])<0: step = (E[-1]-E[-2])/2*(grad[-1]/abs(grad[-1]))
        else: step = 5*(grad[-1]/abs(grad[-1]))
        
    return L, E, grad
