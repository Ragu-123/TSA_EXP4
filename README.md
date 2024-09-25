### NAME :RAGUNATH R
### REGISTER NO:212222240081
### DATE :

# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES


### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the daily minimum temperatures dataset
data = pd.read_csv('/content/daily-minimum-temperatures-in-me.csv')
print(data.head())


# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Extract temperature data and drop any missing values
temperature = data['Daily minimum temperatures'].dropna()

# Simulate ARMA(1,1) process
ar1 = np.array([1, -0.5])  # AR(1) coefficient
ma1 = np.array([1, 0.5])   # MA(1) coefficient
arma11_process = ArmaProcess(ar1, ma1)
arma11_sample = arma11_process.generate_sample(nsample=len(temperature))

# Simulate ARMA(2,2) process
ar2 = np.array([1, -0.5, 0.25])  # AR(2) coefficients
ma2 = np.array([1, 0.4, 0.3])    # MA(2) coefficients
arma22_process = ArmaProcess(ar2, ma2)
arma22_sample = arma22_process.generate_sample(nsample=len(temperature))

# Plot ACF and PACF for the simulated ARMA(1,1) and ARMA(2,2) processes
plt.figure(figsize=(14, 8))

# ACF and PACF for ARMA(1,1)
plt.subplot(221)
plot_acf(arma11_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(1,1)')
plt.subplot(222)
plot_pacf(arma11_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(1,1)')

# ACF and PACF for ARMA(2,2)
plt.subplot(223)
plot_acf(arma22_sample, lags=20, ax=plt.gca(), title='ACF of Simulated ARMA(2,2)')
plt.subplot(224)
plot_pacf(arma22_sample, lags=20, ax=plt.gca(), title='PACF of Simulated ARMA(2,2)')

plt.tight_layout()
plt.show()

````
### OUTPUT:
![image](https://github.com/user-attachments/assets/38a3cac1-341a-4f46-9913-215febfdda12)
![image](https://github.com/user-attachments/assets/d420e23e-a415-481f-9362-f42962524495)
![image](https://github.com/user-attachments/assets/993099c8-b3e6-4d6f-bbd0-e62ad7f16d88)



### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
