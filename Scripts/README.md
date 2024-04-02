# Scripts for Data Analysis Involved in the Project
## Kinematics Data Processing
### Learn Parameters for LDS
[Code](/Scripts/Kinematics_Parameters.py) 

### Perform LDS
Use either manually setted or learnt parameters from the first minute of the mouse foraging data. 
Apply filtering and smoothing functions to the data. 
[Code](/Scripts/Kinematics_LDS.py) 

### Display Smoothed Data
The raw position (x and y), smoothed position (x and y)
and smoothed speed and acceleration are displayed for each session. 
[Code](/Scripts/Kinematics_Display.py) 
![Example for Short Session 0](../Images/Kinematics/ShortSession0.png) 
*The raw and processed mouse kinematics data.*
![Example for Short Session 0](../Images/Positions/ShortSession0.png) 
*The raw and processed mouse position data.*
![Example for Short Session 0](../Figures/LDS.png) 
*A detailed example of the raw and processed (smoothed) mouse position data.*

### Compare Filtering for Manual/Learned Parameters
Compare the filtering results (positions, speed, acceleration)
of either manual or learned parameters. 
[Code](/Scripts/Kinematics_CompareParameters.py) 
![Example for Short Session 0](../Images/CompareParameters/ShortSession0.png) 
*The processed mouse kinematics data of two types of parameters.*

## HMM Models Based on Kinematics Data: Infer the State of the Mouse
### HMM Feature Processing
Get the kinematics data and weight of the mouse in each session. Compute whether mouse
in/out of the two patches based on position data. Each session outputs a dataframe with rows of time index
and columns of position/speed/acceleration/weight/patch etc. that is saved as a .parquet file. 
[Code](/Scripts/HMM_FeatureProcess.py) 

### HMM Model: Determine the Appropriate State Number
Determine the number of states to fit in the GLM-HMM model.
Each short session is used to fit a GLM-HMM model with state number range from 3 to 10. Each model is then used to infer the states of the other short sessions that are concatenated into a complete test session. The loglikelihood of each inference is recorded.
[Code](/Scripts/HMM_FindSingleModel.py) 
![Loglikelihood of Models](../Images/HMM_StateChoice/Summary.png) 
*The summary of the model performances fitted from the 12 short sessions.*

### HMM Model Fitting
Use GLM-HMM model to infer the states of mouse during each session.
The model can be fit separately for each session or using a single session to fit a unified model used for all sessions.
Each session outputs an array of states saved in .npy file.
[Code for Seperated Model Fitting](/Scripts/HMM_FitModels.py) 
[Code for Unified Model Fitting](/Scripts/HMM_FitSingleModel.py) 

### Display Inferred HMM States
Display the inferred states of each session by the unified model. The upper panel is the states in time resolution of
0.2 seconds, and the lower panel is the most-probable state in 10 seconds.
[Code](/Scripts/HMM_States.py) 
![Example for Short Session 0](../Images/HMM_States/ShortSession0.png) 
*The original and down-sampled states of mouse.*

### Display Transition Matrix over Time in Long Sessions
Display the change fo transition matrices over time in long sessions. 
[Code](/Scripts/HMM_LongTransM.py) 

## Regression Models Based on Kinematics Data: Predict Mouse Decisions on Foraging
### Regression Variables Processing
Process the possible dependent variables for predicting each time the distance mouse moves the wheel or
the duration mouse stays within the patch. Regressors include kinematics data, weight, pellets consumed
in the previous visit to the current/the other patch, time since the previous visit to either patch, etc. 
[Code](/Scripts/Regression_FeatureProcess.py) 

### Regression Model Fitting
Use different types of GLM to fit regression models. Models are cross-validated.
[Code](/Scripts/Regression_FitModels.py) 
![Example for All Short Sessions](../Images/Regression/AllSessionsData/Poisson.png) 
*The original and predicted wheel moving distance during each visit of all short sessions, fit by GLM-Poisson model.*
![Example for All Short Sessions](../Images/Regression/AllSessionsData/Gaussian.png) 
*The original and predicted wheel moving distance during each visit of all short sessions, fit by GLM-Gaussian model.*
