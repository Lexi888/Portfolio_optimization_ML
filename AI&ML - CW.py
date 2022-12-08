#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# In[2]:


# Read Data

Multual_Funds = yf.download('TRGRX ADLVX CISMX CFSMX GISYX WTIFX AIAFX BGAIX WAMCX CIPIX GLFOX RYPMX FGILX AASCX FEGIX SGGDX JCMAX VBINX WWWFX BESIX ABALX GEDTX SSSFX WIIFX OEGYX BCSFX FSBBX THISX PVIVX SIUSX', start ='2016-10-01',end ='2022-10-01')


# In[3]:


# Closing price

Multual_Funds = Multual_Funds['Adj Close'] 
print(Multual_Funds)


# ### Calculating the return

# In[4]:


# Log of percentage change of all multual funds in the list

log_return = np.log(Multual_Funds / Multual_Funds.shift(1))


# In[5]:


log_return.tail().round(3)


# ### Calculating the Mean and Standard deviation

# In[6]:


### Calculating the correlation and covariance

Correlation_matrix = log_return.corr()


# In[7]:


pd.options.display.max_columns = None


# In[8]:


Correlation_matrix


# In[9]:


### Calculating the covariance

Covariance_matrix = log_return.cov()
Covariance_matrix


# In[10]:


# Read Data

Multual_Funds_Selected = yf.download('SIUSX FEGIX CISMX BCSFX WWWFX', start ='2016-10-01',end ='2022-10-01')
Multual_Funds_Selected['Adj Close'].tail()


# In[11]:


Multual_Funds_Selected = Multual_Funds_Selected['Adj Close']


# In[12]:


# Log of percentage change of selected multual funds in the list

log_return_selected = np.log(Multual_Funds_Selected / Multual_Funds_Selected.shift(1))


# In[13]:


### Calculating the correlation for selected multual funds

Correlated_matrix_selected = log_return_selected.corr()
Correlated_matrix_selected


# In[15]:


# Calculating the covariance for selected multual funds

Covariance_matrix_selected = log_return_selected.cov()
Covariance_matrix_selected


# In[16]:


# Equally weighted portfolio's variance

w = {'SIUSX':0.2, 'FEGIX':0.2, 'CISMX':0.2, 'BCSFX':0.2,'WWWFX':0.2}
portfolio_var = Covariance_matrix_selected.mul(w, axis=0).mul(w, axis=1).sum().sum()
Ann_portfolio_std = (portfolio_var*252)**(0.5)
print(portfolio_var)
print(portfolio_var*252)


# In[17]:


# Yearly returns for individual companies

individual_return = Multual_Funds_Selected.resample('Y').last().pct_change().mean() 
individual_return


# In[18]:


# Portfolio expected returns

w = [0.2, 0.2, 0.2, 0.2,0.2] 
portfolio_return = (w*individual_return).sum()
portfolio_return


# In[19]:


# Calculate Sharpe ratio
rf = 0.01
Sharpe_ratio = (portfolio_return-rf)/Ann_portfolio_std 
print(Sharpe_ratio)


# ### ML for portfolio construction

# In[20]:


port_return = [] 
# Define an empty array for portfolio returns 
port_vol = [] 
# Define an empty array for portfolio volatility 
port_weights = [] 
# Define an empty array for asset weights
port_sharpe = []
# Define an empty array for sharpe ratio
num_assets = len(Multual_Funds_Selected.columns) 
num_portfolios = 10000

# Simulate 10000 times


# In[21]:


for portfolio in range(num_portfolios): 
    weights = np.random.random(num_assets)
    # weights generated randomly
    
    weights = weights/np.sum(weights)
    port_weights.append(weights)
    returns = np.dot(weights, individual_return) 
    # Returns are the product of individual expected returns of asset and its # weights
    
    port_return.append(returns)
    var = Covariance_matrix_selected.mul(weights, axis=0).mul(weights, axis=1).sum().sum() 
    # Portfolio Variance 
    
    sd = np.sqrt(var) 
    #Daily standard deviation
    ann_sd = sd*np.sqrt(252) 
    #Annual standard deviation = volatility 
    Sharpe_ratio1 = (returns-rf)/ann_sd
    port_sharpe.append(Sharpe_ratio1)
    port_vol.append(ann_sd)

data = {'Returns':port_return, 'Volatility':port_vol, 'Sharpe ratio':port_sharpe}
for counter, symbol in enumerate(Multual_Funds_Selected.columns.tolist()): 
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in port_weights]
portfolios = pd.DataFrame(data)
portfolios.head() 


# In[22]:


# Plot efficient frontier
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[5,5])


# In[23]:


min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()] 
# idxmin() gives us the minimum value in the column specified. 
min_vol_port
# For minimum volatility, the weights are assigned below:


# In[24]:


# plotting the minimum volatility portfolio

plt.subplots(figsize=[6,5])
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3) 
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=200)


# In[25]:


# Finding the optimal portfolio
rf = 0.01 
# risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()] 
optimal_risky_port


# In[26]:


# Plotting optimal portfolio

plt.subplots(figsize=(5, 6))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3) 
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500) 
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)


# In[28]:


port_sharpe_mean = portfolios['Sharpe ratio'].mean()
port_sharpe_mean

