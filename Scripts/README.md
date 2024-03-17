# Scripts for Data Analysis Involved in the Project
## Kinematics Data Processing
### LDS
Use either manually setted or learnt parameters from the first minute of the mouse foraging data. 
Apply filtering and smoothing functions to the data. 
[Code](/Scripts/Kinematics_Processing.py) 

### Display Smoothed Data
The raw position (x and y), smoothed position (x and y)
and smoothed speed and acceleration is displayed for each session. 
[Code](/Scripts/Kinematics_Display.py) 
![Example for Short Session 0](../Images/Kinematics/ShortSession0.png)  
*The raw and processed mouse kinematics data.*

### Compare Filtering for Manual/Learned Parameters
Compare the filtering results (positions, speed, acceleration)
of either manual or learned parameters.
[Code](/Scripts/Kinematics_CompareParameters.py) 
![Example for Short Session 0](../Images/CompareParameters/ShortSession0.png)  
*The processed mouse kinematics data of two types of parameters.*
