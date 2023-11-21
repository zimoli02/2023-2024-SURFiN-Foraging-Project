# Notes and Figures Reproduced from 'Time Series Analysis by State Space Methods' (Durbin and Koopman)
## Figure 2.1
![Figure2.1](Fig_2.1.png)  
*Figure 2.1: Nile data and output of Kalman filter: (i) data (dots), filtered state at (solid line) and its 90% confidence intervals (light solid lines); (ii) filtered state variance $P_t$; (iii) prediction errors $v_t$; (iv) prediction variance $F_t$.*

![Figure2.1_KalmanGain](Fig_2.1_KalmanGain.png)  
*Kalman gain $K_t$, converging to a steady state.*

![Figure2.1_SteadyState](Fig_2.1_SteadyState.png)  
*An updated version of Figure 2.1, using a steady-state Kalman gain when its difference with the online Kalman gain is less than $1e3$.*  

Code for the three figures above can be found [here](Figure_2.1.py)

## Figure 2.2
![Figure2.2](Fig_2.2.png)  
*Figure 2.2: Nile data and output of state smoothing recursion: (i) data (dots), smoothed state $αˆt$ and its 90% confidence intervals; (ii) smoothed state variance $v_t$; (iii) smoothing cumulant $r_t$; (iv) smoothing variance cumulant $N_t$.*  
Code for the figure can be found [here](Figure_2.2.py)

## Figure 2.5
![Figure2.5](Fig_2.5.png)  
*Figure 2.5: Filtering and smoothing output when observations are missing: (i) data and filtered state at (extrapolation); (ii) filtered state variance $P_t$; (iii) data and smoothed state $αˆt$ (interpolation); (iv) smoothed state variance $V_t$.*  
Code for the figure can be found [here](Figure_2.5.py)

## Figure 2.6
![Figure2.6](Fig_2.6.png)  
*Figure 2.6: Nile data and output of forecasting: (i) data (dots), state forecast at and 50% confidence intervals; (ii) state variance $P_t$; (iii) observation forecast $E(y_t|Y_t−1)$; (iv) observation forecast variance $F_t$.*  
Code for the figure can be found [here](Figure_2.6.py)

## Figure 2.7
![Figure2.7](Fig_2.7.png)  
*Figure 2.7: Diagnostic plots for standardised prediction errors: (i) standardised residual; (ii) histogram plus estimated density; (iii) ordered residuals; (iv) correlogram.*  
Code for the figure can be found [here](Figure_2.7.py)

If the observation is not linearly related to the state but belongs to a Poisson distribution $Poisson(\lambda = exp(x_n))$, its fitness with the Kalman filter can be assessed by:  
![Figure2.7_Poisson](Fig_2.7_PoissonData.png)   
Code for the figure can be found [here](Figure_2.7_Poisson.py)

where:   
$S =  2.55, K =  15.39, N =  739.53, H(33) =  0.08, Q(9) =  45.91$  
$p_S=  0.0 ,  p_K=0.0 $


If the observation noise is acculumated with time, i.e.:  
$y_t = x_t + \eta_t$  
$\eta_t = \eta_{t-1} + \omega_t$  
$\omega_t \sim N(0, \sigma^2_\omega)$

Its fitness with the Kalman filter can be assessed by:  
![Figure2.7_Noise](Fig_2.7_CorrelateNoise.png)   
Code for the figure can be found [here](Figure_2.7_CorrelateNoise.py)

Then, the simulated data and observation noise for 100 realizations would be:  
![CorrelateNoiseSimulation](CorrelatedNoise.png)  
Code for the figure can be found [here](CorrelatedNoise.py)

In this circumstance, the observation noise $\eta_t$ is heteroscedastic:  
$Var[\eta_t]=Var[\eta_{t-1}]+Var[\omega_t]=Var[\eta_0] + \sum_{i=1}^tVar[\omega_i]=t\sigma^2_\omega$

The observation noises are correlated:  
$Cov[\eta_n, \eta_m]=Cov[\sum_{i=1}^n\omega_i, \sum_{i=1}^m\omega_i]$  

Because each $\omega_i$ and $\omega_j$ is independent:  
$Cov[\omega_i,\omega_j]=E[(\omega_i-E[\omega_i])(\omega_j - E[\omega_j])]=E[\omega_i\omega_j]=E[\omega_i]E[\omega_j]=0$  

Hence:  
$Cov[\eta_n, \eta_m]=Cov[\sum_{i=1}^n\omega_i, \sum_{i=1}^m\omega_i]=min(n,m)*\sigma^2_\omega$