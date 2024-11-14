
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import random

import statsmodels.api as sm
import scipy.stats as stats


def pcaWeights(cov, riskDist=None, riskTarget=1.):
    eVal, eVec = np.linalg.eig(cov)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:, indices]
    if riskDist is None:
        riskDist = np.zeros(cov.shape[0])
        riskDist[-1] = 1
    loads = riskTarget * (riskDist/eVal)**.5
    wghts = np.dot(eVec, loads.reshape(-1, 1))
    return wghts

def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sPos > h:
            sPos = 0
            tEvents.append(i)
        if sNeg < -h:
            sNeg = 0
            tEvents.append(i)
    return tEvents

def advanced_bars(series, h, bar_type = 'Volume'):
    index_name = series.index.name
    df = pd.DataFrame()
    temp_data = series.reset_index()
    current_vol = 0
    for i in range(len(temp_data)):
        if current_vol == 0:
                date, cum_high,cum_low,cum_open,cum_close,cum_adj_close = temp_data.iloc[i][index_name],temp_data.iloc[i]['High'],temp_data.iloc[i]['Low'],temp_data.iloc[i]['Open'],temp_data.iloc[i]['Close'],temp_data.iloc[i]['Adj Close']
        cum_high, cum_low = max(cum_high, temp_data.iloc[i]['High']), min(cum_low, temp_data.iloc[i]['Low'])
        if bar_type == 'Volume':
            current_vol += temp_data.iloc[i]['Volume']
        elif bar_type == 'Dollar':
            current_vol += temp_data.iloc[i]['Volume'] * temp_data.iloc[i]['Close']
        elif bar_type == 'Tick':
            current_vol += 1
        else:
            raise ValueError("bar_type Wrong! Please enter correct bar_type: Volume, Dollar or Tick")

        if (current_vol >= h):
            cum_close = temp_data.iloc[i][ 'Close']
            cum_adj_close = temp_data.iloc[i]['Adj Close']
            temp = pd.DataFrame([[date, cum_adj_close, cum_close, cum_high, cum_low, cum_open, current_vol]], columns = [index_name,'Adj Close','Close','High','Low','Open','Cum '+bar_type])
            current_vol = 0
            df = pd.concat([df, temp], axis=0)
    df.set_index(index_name, inplace=True)
    return df

def calculate_imbalance_bars(df, initial_threshold, ewma_alpha=0.95, bar_type = 'Dollar'):
    """
    Calculate dollar imbalance bars with an EWMA-adjusted threshold as per de Prado's method.
    
    Parameters:
    df (pd.DataFrame): A DataFrame with columns ['date', 'open', 'close', 'high', 'low', 'volume'].
    initial_threshold (float): Initial imbalance threshold for triggering a bar.
    ewma_alpha (float): Smoothing factor for EWMA (e.g., 0.95 gives strong weighting to recent observations).
    
    Returns:
    pd.DataFrame: A DataFrame containing the dollar imbalance bars.
    """
    name = df.index.name
    df = df.reset_index()
    imbalance_bars = []  # Store completed imbalance bars
    cumulative_signed_flow = 0  # Tracks cumulative signed imbalance flow
    ewma_threshold = initial_threshold  # Set initial threshold
    Date, Open, High, Low, Volume =df[name].iloc[0], df['Open'].iloc[0], df['High'].iloc[0], df['Low'].iloc[0], df['Volume'].iloc[0]
    i_prev = 0
    T_array = []
    imbalance_array = []
    
    for i in range(1, len(df)):
        price_change = df['Close'].iloc[i] - df['Close'].iloc[i - 1]
        tick_direction = np.sign(price_change)  # Determine tick direction
        if bar_type == 'Dollar':
            signed_flow = tick_direction * df['Volume'].iloc[i] * df['Close'].iloc[i]  # Calculate signed flow
        elif bar_type == 'Volume':
            signed_flow = tick_direction * df['Volume'].iloc[i]
        elif bar_type == 'Tick':
            signed_flow = tick_direction * 1
        else:
            raise ValueError("Wrong bar_type, please select the correct type: Dollar, Volume, Tick")
        
        cumulative_signed_flow += signed_flow  # Accumulate signed flow
        imbalance_array.append(signed_flow)
        
        Volume += df['Volume'].iloc[i]
        High = max(High, df['High'].iloc[i])
        Low = min(Low, df['Low'].iloc[i])

        # Check if we should create a new bar
        if abs(cumulative_signed_flow) >= ewma_threshold:
            # Add the current bar's details to the bars list
            bar = {
                name: Date,
                'Open': Open,
                'Close': df['Close'].iloc[i],
                'High': High,
                'Low': Low,
                'Volume': Volume,
                'Adj Close': df['Adj Close'].iloc[i],
                'Cumulative_signed_flow': cumulative_signed_flow,
                'Threshold': ewma_threshold
            }
            if (i < len(df)-1):
                Open = df['Open'].iloc[i + 1]
                High = df['High'].iloc[i + 1]
                Low = df['Low'].iloc[i + 1]
                Date = df[name].iloc[i + 1]
            i_prev = i
            T_array.append(i - i_prev + 1)
            
            imbalance_bars.append(bar)
            
            # Update cumulative imbalance and EWMA threshold
            E_imbalance = pd.Series(imbalance_array).ewm(alpha=ewma_alpha).mean().values[-1]
            E_T = pd.Series(T_array).ewm(alpha=ewma_alpha).mean().values[-1]
            ewma_threshold = E_T * E_imbalance
            # Reset cumulative signed flow for the next bar
            cumulative_signed_flow = 0
            Volume = 0

    # Return result as a DataFrame
    imbalance_bars_df = pd.DataFrame(imbalance_bars)
    return imbalance_bars_df
