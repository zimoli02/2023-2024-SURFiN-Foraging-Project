import math
import time
import numpy as np
import scipy.optimize
import torch
import warnings
import copy

import utils

import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
parent_dir = current_script_path.parent.parent
sys.path.insert(0, str(parent_dir))

import Functions.inference as inference


iteration = 0


def scipy_optimize_SS_tracking_fullV0(y, B, sigma_a0, Qe, Z, diag_R_0,
                                      m0_0, V0_0, max_iter, disp=True):
    iL_V0 = np.tril_indices(V0_0.shape[0])

    def get_coefs_from_params(sigma_a, diag_R, m0, V0, iL_V0=iL_V0):
        V0_chol = np.linalg.cholesky(V0)
        L_coefs = V0_chol[iL_V0]
        x = np.insert(np.concatenate([diag_R, m0, L_coefs]), 0, sigma_a)
        return x

    def get_params_from_coefs(x, iL_V0=iL_V0, sigma_a0=sigma_a0,
                              diag_R_0=diag_R_0, m0_0=m0_0, V0_0=V0_0):
        cur = 0
        sigma_a = x[slice(cur, cur+1)]
        cur += len(sigma_a)
        diag_R = x[slice(cur, cur+len(diag_R_0))]
        cur += len(diag_R)
        m0 = x[slice(cur, cur+len(m0_0))]
        cur += len(m0)
        M = V0_0.shape[0]
        L_coefs = x[slice(cur, cur+int(M*(M+1)/2))]
        L_V0 = np.zeros(shape=V0_0.shape)
        L_V0[iL_V0] = L_coefs
        V0 = L_V0 @ L_V0.T

        return sigma_a, diag_R, m0, V0

    def optim_criterion(x):
        sigma_a, diag_R, m0, V0 = get_params_from_coefs(x)
        R = np.diag(diag_R)
        kf = inference.filterLDS_SS_withMissingValues(y=y, B=B, Q=sigma_a*Qe,
                                                      m0=m0, V0=V0, Z=Z, R=R)
        answer = 0
        N = kf["Sn"].shape[2]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:,:,n] 

            Sn_inv = np.linalg.inv(Sn)
            answer += np.linalg.slogdet(Sn)[1]
            answer += innov.T @ Sn_inv @ innov
        return answer

    def callback(x):
        global iteration
        iteration += 1

        sigma_a, diag_R, m0, V0 = get_params_from_coefs(x)
        optim_value = optim_criterion(x=x)
        print("Iteration: {:d}".format(iteration))
        print("optim criterion: {:f}".format(optim_value.item()))
        print("sigma_a={:f}".format(sigma_a.item()))
        print("diag_R:")
        print(diag_R)
        print("m0:")
        print(m0)
        print("V0:")
        print(V0)

    x0 = get_coefs_from_params(sigma_a=sigma_a0, diag_R=diag_R_0,
                               m0=m0_0, V0=V0_0)
    options={"disp": disp, "maxiter": max_iter}
    opt_res = scipy.optimize.minimize(optim_criterion, x0, method="Nelder-Mead",
                                      callback=callback, options=options)
    import pdb; pdb.set_trace()


def scipy_optimize_SS_tracking_diagV0(y, B, sigma_ax0, sigma_ay0, Qe, Z,
                                      sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                      max_iter=50, disp=True):

    def get_coefs_from_params(sigma_ax, sigma_ay, sqrt_diag_R, m0,
                              sqrt_diag_V0):
        x = np.concatenate([[sigma_ax, sigma_ay], sqrt_diag_R, m0,
                            sqrt_diag_V0])
        return x

    def get_params_from_coefs(x, sigma_ax0=sigma_ax0, sigma_ay0=sigma_ay0,
                              sqrt_diag_R_0=sqrt_diag_R_0, m0_0=m0_0,
                              sqrt_diag_V0_0=sqrt_diag_V0_0):
        cur = 0
        sigma_ax = x[slice(cur, cur+1)]
        cur += len(sigma_ax)
        sigma_ay = x[slice(cur, cur+1)]
        cur += len(sigma_ay)
        sqrt_diag_R = x[slice(cur, cur+len(sqrt_diag_R_0))]
        cur += len(sqrt_diag_R)
        m0 = x[slice(cur, cur+len(m0_0))]
        cur += len(m0)
        sqrt_diag_V0 = x[slice(cur, cur+len(sqrt_diag_V0_0))]

        return sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0

    def optim_criterion(x):
        sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
            get_params_from_coefs(x)
        V0 = np.diag(sqrt_diag_V0**2)
        R = np.diag(sqrt_diag_R**2)
        # build Q from Qe, sigma_ax, sigma_ay
        Q = utils.buildQfromQe_np(Qe=Qe, sigma_ax=sigma_ax, sigma_ay=sigma_ay)

        kf = inference.filterLDS_SS_withMissingValues(y=y, B=B, Q=Q,
                                                      m0=m0, V0=V0, Z=Z, R=R)
        answer = 0
        N = kf["Sn"].shape[2]
        for n in range(N):
            innov = kf["innov"][:, :, n]
            Sn = kf["Sn"][:, :, n]

            Sn_inv = np.linalg.inv(Sn)
            answer += np.linalg.slogdet(Sn)[1]
            answer += innov.T @ Sn_inv @ innov
        return answer

    def callback(x):
        global iteration
        iteration += 1

        sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
            get_params_from_coefs(x)
        optim_value = optim_criterion(x=x)
        print("Iteration: {:d}".format(iteration))
        print("optim criterion: {:f}".format(optim_value.item()))
        print("sigma_ax={:f}".format(sigma_ax.item()))
        print("sigma_ay={:f}".format(sigma_ay.item()))
        print("sqrt_diag_R:")
        print(sqrt_diag_R)
        print("m0:")
        print(m0)
        print("sqrt_diag_V0:")
        print(sqrt_diag_V0)

    x0 = get_coefs_from_params(sigma_ax=sigma_ax0, sigma_ay=sigma_ay0,
                               sqrt_diag_R=sqrt_diag_R_0, m0=m0_0,
                               sqrt_diag_V0=sqrt_diag_V0_0)
    options = {"disp": disp, "maxiter": max_iter}
    opt_res = scipy.optimize.minimize(optim_criterion, x0,
                                      method="Nelder-Mead", callback=callback,
                                      options=options)
    sigma_ax, sigma_ay, sqrt_diag_R, m0, sqrt_diag_V0 = \
        get_params_from_coefs(opt_res["x"])
    x = {"sigma_ax": sigma_ax, "sigma_ay": sigma_ay,
         "sqrt_diag_R": sqrt_diag_R, "m0": m0,
         "sqrt_diag_V0": sqrt_diag_V0}
    answer = {"fun": opt_res["fun"], "message": opt_res["message"],
              "nfev": opt_res["nfev"], "nit": opt_res["nit"],
              "status": opt_res["status"], "success": opt_res["success"],
              "x": x}
    return answer

def torch_lbfgs_optimize_SS_tracking_diagV0(y, B, sigma_a0, Qe, Z,
                                            sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                            max_iter=20, lr=1.0,
                                            tolerance_grad=1e-7,
                                            tolerance_change=1e-9,
                                            n_epochs = 100, tol=1e-6,
                                            line_search_fn="strong_wolfe",
                                            vars_to_estimate={
                                                "sigma_a": True,
                                                "R": True, "m0": True, "V0": True},
                                            disp=True):

    def log_likelihood_fn():
        V0 = torch.diag(sqrt_diag_V0**2)
        R = torch.diag(sqrt_diag_R**2)
        Q = Qe * sigma_a**2
        log_like = inference.logLikeLDS_SS_withMissingValues_torch(
            y=y, B=B, Q=Q, m0=m0, V0=V0, Z=Z, R=R)
        return log_like

    optim_params = {"max_iter": max_iter, "lr": lr,
                    "tolerance_grad": tolerance_grad,
                    "tolerance_change": tolerance_change,
                    "line_search_fn": line_search_fn}
    sigma_a = torch.tensor([sigma_a0], dtype=torch.double)
    sqrt_diag_R = sqrt_diag_R_0
    m0 = m0_0
    sqrt_diag_V0 = sqrt_diag_V0_0
    x = []
    if vars_to_estimate["sigma_a"]:
        x.append(sigma_a)
    if vars_to_estimate["sqrt_diag_R"]:
        x.append(sqrt_diag_R)
    if vars_to_estimate["m0"]:
        x.append(m0)
    if vars_to_estimate["sqrt_diag_V0"]:
        x.append(sqrt_diag_V0)
    if len(x) == 0:
        raise RuntimeError("No variable to estimate. Please set one element "
                           "of vars_to_estimate to True")
    optimizer = torch.optim.LBFGS(x, **optim_params)
    for i in range(len(x)):
        x[i].requires_grad = True

    def closure():
        optimizer.zero_grad()
        curEval = -log_likelihood_fn()
        print(f"in closure, ll={-curEval}")
        curEval.backward(retain_graph=True)
        return curEval

    termination_info = "success: reached maximum number of iterations"
    log_like = []
    elapsed_time = []
    start_time = time.time()
    for epoch in range(n_epochs):
        prev_x = copy.deepcopy(x)
        try:
            curEval = optimizer.step(closure)
        except RuntimeError:
            # begin backtracking
            if vars_to_estimate["sigma_a"]:
                sigma_a = prev_x.pop(0)
                sigma_a.requires_grad = False
            if vars_to_estimate["sqrt_diag_R"]:
                sqrt_diag_R = prev_x.pop(0)
                sqrt_diag_R.requires_grad = False
            if vars_to_estimate["m0"]:
                m0 = prev_x.pop(0)
                m0.requires_grad = False
            if vars_to_estimate["sqrt_diag_V0"]:
                sqrt_diag_V0 = prev_x.pop(0)
                sqrt_diag_V0.requires_grad = False
            # end backtracking
            termination_info = "nan generated"
            break
        log_like.append(-curEval.item())
        elapsed_time.append(time.time() - start_time)
        print("--------------------------------------------------------------------------------")
        print(f"epoch: {epoch}")
        print(f"likelihood: {log_like[-1]}")
        if vars_to_estimate["sigma_a"]:
            print("sigma_a: ")
            print(sigma_a)
        if vars_to_estimate["sqrt_diag_R"]:
            print("sqrt_diag_R: ")
            print(sqrt_diag_R)
        if vars_to_estimate["m0"]:
            print("m0: ")
            print(m0)
        if vars_to_estimate["sqrt_diag_V0"]:
            print("sqrt_diag_V0: ")
            print(sqrt_diag_V0)
        if epoch > 0 and log_like[-1] - log_like[-2] < tol:
            termination_info = "success: converged"
            break
    for i in range(len(x)):
        x[i].requires_grad = False

    estimates = {}
    if vars_to_estimate["sigma_a"]:
        estimates["sigma_a"] = sigma_a
    if vars_to_estimate["sqrt_diag_R"]:
        estimates["sqrt_diag_R"] = sqrt_diag_R
    if vars_to_estimate["m0"]:
        estimates["m0"] = m0
    if vars_to_estimate["sqrt_diag_V0"]:
        estimates["sqrt_diag_V0"] = sqrt_diag_V0
    answer = {"estimates": estimates,
              "log_like": log_like,
              "elapsed_time": elapsed_time,
              "termination_info": termination_info}
    return answer

def torch_adam_optimize_SS_tracking_diagV0(y, B, sigma_a0, Qe, Z,
                                           sqrt_diag_R_0, m0_0, sqrt_diag_V0_0,
                                           max_iter=50, lr=1e-3, eps=1e-8,
                                           vars_to_estimate={
                                               "sigma_a": True,
                                               "R": True, "m0": True,
                                               "V0": True},
                                          ):

    def log_likelihood_fn():
        V0 = torch.diag(sqrt_diag_V0**2)
        R = torch.diag(sqrt_diag_R**2)
        Q = Qe * sigma_a**2
        kf = inference.filterLDS_SS_withMissingValues_torch(y=y, B=B, Q=Q,
                                                            m0=m0, V0=V0, Z=Z,
                                                            R=R)
        log_like = kf["logLike"]
        return log_like

    optim_params = {"lr": lr, "eps": eps}
    sigma_a = torch.Tensor([sigma_a0])
    sqrt_diag_R = sqrt_diag_R_0
    m0 = m0_0
    sqrt_diag_V0 = sqrt_diag_V0_0
    x = []
    if vars_to_estimate["sigma_a"]:
        x.append(sigma_a)
    if vars_to_estimate["sqrt_diag_R"]:
        x.append(sqrt_diag_R)
    if vars_to_estimate["m0"]:
        x.append(m0)
    if vars_to_estimate["sqrt_diag_V0"]:
        x.append(sqrt_diag_V0)
    if len(x) == 0:
        raise RuntimeError("No variable to estimate. Please set one element "
                           "of vars_to_estimate to True")
    optimizer = torch.optim.Adam(x, **optim_params)
    for i in range(len(x)):
        x[i].requires_grad = True

    log_like = []
    elapsed_time = []
    start_time = time.time()
    for i in range(max_iter):
        breakpoint()
        optimizer.zero_grad()
        curEval = -log_likelihood_fn()
        curEval.backward()
        optimizer.step()
        log_like.append(-curEval.item())
        elapsed_time.append(time.time() - start_time)
        print("--------------------------------------------------------------------------------")
        print(f"iteration: {i} ({max_iter})")
        print(f"logLike: {-curEval}")
        print("sigma_a: ")
        print(sigma_a)
        print("sqrt_diag_R: ")
        print(sqrt_diag_R)
        print("m0: ")
        print(m0)
        print("sqrt_diag_V0: ")
        print(sqrt_diag_V0)

    for i in range(len(x)):
        x[i].requires_grad = False

    estimates = {}
    if vars_to_estimate["sigma_a"]:
        e_sigma_a = x.pop(0)[0].item()
        estimates["sigma_a"] = e_sigma_a
    if vars_to_estimate["sqrt_diag_R"]:
        e_sqrt_diag_R = x.pop(0)
        estimates["sqrt_diag_R"] = e_sqrt_diag_R
    if vars_to_estimate["m0"]:
        e_m0 = x.pop(0)
        estimates["m0"] = e_m0
    if vars_to_estimate["sqrt_diag_V0"]:
        e_sqrt_diag_V0 = x.pop(0).numpy()
        estimates["sqrt_diag_V0"] = e_sqrt_diag_V0
    answer = {"estimates": estimates,
              "log_like": log_like,
              "elapsed_time": elapsed_time,
             }
    return answer


def em_SS_tracking(y, B, sigma_a0, Qe, Z, R_0, m0_0, V0_0,
                   vars_to_estimate={"sigma_a": True, "R": True,
                                     "m0": True, "V0": True},
                   max_iter=50, tolerance_change=1e-9):
    sigma_a = sigma_a0
    R = R_0
    m0 = m0_0
    V0 = V0_0

    Qe_inv = np.linalg.inv(Qe)
    N = y.shape[1]
    M = Qe.shape[0]
    termination_info = "success: reached maximum number of iterations"
    log_like = []
    elapsed_time = []
    start_time = time.time()
    for iter in range(max_iter):
        # E step
        Q = Qe * sigma_a**2
        kf = inference.filterLDS_SS_withMissingValues_np(y=y, B=B,
                                                         Q=Q, m0=m0, V0=V0,
                                                         Z=Z, R=R)
        print("LogLike[{:04d}]={:f}".format(iter, kf["logLike"].item()))
        print("sigma_a={:f}".format(sigma_a))
        print("R:")
        print(R)
        print("m0:")
        print(m0)
        print("V0:")
        print(V0)
        log_like.append(kf["logLike"].item())
        if math.isnan(log_like[-1]) or iter > 0 and log_like[-1] < log_like[-2]:
            if math.isnan(log_like[-1]):
                termination_info = "nan detected"
            else:
                termination_info = "log likelihood increased"
            # begin backtrack
            if vars_to_estimate["sigma_a"]:
                sigma_a = sigma_a_prev
            if vars_to_estimate["R"]:
                R = R_prev
            if vars_to_estimate["m0"]:
                m0 = m0_prev
            if vars_to_estimate["V0"]:
                V0 = V0_prev
            del log_like[-1]
            # end backtrack
            break
        elif iter > 0 and log_like[-1] - log_like[-2] < tolerance_change:
            termination_info = "converged"
            break
        elapsed_time.append(time.time() - start_time)
        ks = inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Vnn=kf["Vnn"],
                                    xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                    m0=m0, V0=V0)
        # M step
        if vars_to_estimate["sigma_a"]:
            sigma_a_prev = sigma_a
            S11, S10, S00 = posteriorCorrelationMatrices(Z=Z, B=B, KN=kf["KN"],
                                                         Vnn=kf["Vnn"], xnN=ks["xnN"],
                                                         VnN=ks["VnN"], x0N=ks["x0N"],
                                                         V0N=ks["V0N"], Jn=ks["Jn"],
                                                         J0=ks["J0"])
            # sigma_a
            W = S11 - S10 @ B.T - B @ S10.T + B @ S00 @ B.T
            U = W @ Qe_inv
            K = np.trace(U)
            sigma_a = np.sqrt(K/(N*M))
        # R
        if vars_to_estimate["R"]:
            R_prev = R
            u = y[:, 0] - (Z @ ks["xnN"][:, :, 0]).squeeze()
            R = np.outer(u, u) + Z @ ks["VnN"][:, :, 0] @ Z.T
            for i in range(1, N):
                u = y[:, i] - (Z @ ks["xnN"][:, :, i]).squeeze()
                R = R + np.outer(u, u) + Z @ ks["VnN"][:, :, i] @ Z.T
            R = R/N

        # m0, V0
        if vars_to_estimate["m0"]:
            m0_prev = m0
            m0 = ks["x0N"].squeeze()

        if vars_to_estimate["V0"]:
            V0_prev = V0
            V0 = ks["V0N"]

    estimates = {}
    if vars_to_estimate["sigma_a"]:
        estimates["sigma_a"] = sigma_a
    if vars_to_estimate["R"]:
        estimates["R"] = R
    if vars_to_estimate["m0"]:
        estimates["m0"] = m0
    if vars_to_estimate["V0"]:
        estimates["V0"] = V0

    optim_res = {"estimates": estimates,
                 "log_like": log_like,
                 "elapsed_time": elapsed_time,
                 "termination_info": termination_info,
                }
    return optim_res


def em_SS_LDS(y, B0, Q0, Z0, R0, m0, V0, max_iter=50, tol=1e-4,
              varsToEstimate=dict(m0=True, V0=True, B=True, Q=True, Z=True,
                                  R=True)):
    B  = B0
    Q  = Q0
    Z  = Z0
    R  = R0
    V0 = V0

    M = B0.shape[0]
    N = y.shape[1]
    log_like = np.empty(max_iter)

    for iter in range(max_iter):
        kf = inference.filterLDS_SS_withMissingValues_np(y=y, B=B,
                                                         Q=Q, m0=m0, V0=V0,
                                                         Z=Z, R=R)
        print("LogLike[{:04d}]={:f}".format(iter, kf["logLike"].item()))
        log_like[iter] = kf["logLike"]
        ks = inference.smoothLDS_SS(B=B, xnn=kf["xnn"], Vnn=kf["Vnn"],
                                    xnn1=kf["xnn1"], Vnn1=kf["Vnn1"],
                                    m0=m0, V0=V0)

        S11, S10, S00 = posteriorCorrelationMatrices(Z=Z, B=B, KN=kf["KN"],
                                                     Vnn=kf["Vnn"], xnN=ks["xnN"],
                                                     VnN=ks["VnN"], x0N=ks["x0N"],
                                                     V0N=ks["V0N"], Jn=ks["Jn"],
                                                     J0=ks["J0"])
        if varsToEstimate["Z"]:
            Z = np.outer(y[:,0], ks["xnN"][:, :, 0])
            for i in range(1, N):
                Z = Z + np.outer(y[:, i], ks["xnN"][:, :, i])
            Z = Z @ np.linalg.inv(S11)

        if varsToEstimate["B"]:
            B = S10 @ np.linalg.inv(S00)

        if varsToEstimate["Q"]:
            Q = (S11 - S10 @ np.linalg.inv(S00) @ S10.T)/N
            Q = (Q.T + Q)/2

        # Now that Z is estimated, lets estimate R, if requested
        if varsToEstimate["R"]:
            u = y[:, 0] - (Z @ ks["xnN"][:, :, 0]).squeeze()
            R = np.outer(u, u) + Z @ ks["VnN"][:, :, 0] @ Z.T
            for i in range(1, N):
                u = y[:, i] - (Z @ ks["xnN"][:, :, i]).squeeze()
                R = R + np.outer(u, u) + Z @ ks["VnN"][:, :, i] @ Z.T
            R = R/N

        if varsToEstimate["m0"]:
            m0 = ks["x0N"]

        if varsToEstimate["V0"]:
            V0 = ks["V0N"]

    answer = dict(B=B, Q=Q, Z=Z, R=R, m0=m0, V0=V0, log_like=log_like[:iter],
                  niter=iter)
    return answer

def posteriorCorrelationMatrices(Z, B, KN, Vnn, xnN, VnN, x0N, V0N, Jn, J0):
    # We want to first estimate Z and then R, because R depends on Z
    Vnn1N = lag1CovSmootherLDS_SS(Z=Z, KN=KN, B=B, Vnn=Vnn, Jn=Jn, J0=J0)
    S11 = np.outer(xnN[:,:,0], xnN[:,:,0]) + VnN[:,:,0]
    S10 = np.outer(xnN[:,:,0], x0N) + Vnn1N[:,:,0]
    S00 = np.outer(x0N, x0N) + V0N
    N = xnN.shape[2]
    for i in range(1, N):
        S11 = S11 + np.outer(xnN[:, :, i], xnN[:, :, i]) + VnN[:, :, i]
        S10 = S10 + np.outer(xnN[:, :, i], xnN[:, :, i-1]) + Vnn1N[:, :, i]
        S00 = S00 + np.outer(xnN[:, :, i-1], xnN[:, :, i-1]) + VnN[:, :, i-1]
    return S11, S10, S00


def lag1CovSmootherLDS_SS(Z, KN, B, Vnn, Jn, J0):
    #SS16, Property 6.3
    M = KN.shape[0]
    N = Vnn.shape[2]
    Vnn1N = np.empty(shape=(M, M, N))
    eye = np.eye(M)
    Vnn1N[:, :, N-1] = (eye - KN @ Z) @ B @ Vnn[:, :, N-2]
    for k in range(N-1, 1, -1):
        Vnn1N[:, :, k-1] = Vnn[:, :, k-1] @ Jn[:, :, k-2].T + \
                           Jn[:, :, k-1] @ \
                           (Vnn1N[:, :, k] - B @ Vnn[:, :, k-1]) @ Jn[:, :, k-2].T
    Vnn1N[:, :, 0] = Vnn[:, :, 0] @ J0.T + Jn[:, :, 0] @ (Vnn1N[:, :, 1] - B @ Vnn[:, :, 0]) @ J0.T
    return Vnn1N