import sys
print(sys.version)
# For some reason Theano is unhappy when I run the GP, need to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import requests
import pandas as pd
import numpy as np
import scipy.stats as sps
from scipy.special import factorial
import math
import scipy.special as spsp

#from matplotlib import pyplot as plt
#from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
#from matplotlib.dates import date2num
#import matplotlib.dates as mdates
#import matplotlib.ticker as ticker

from datetime import date
from datetime import datetime
from datetime import timedelta

FILTERED_REGION_CODES = ['']
#FILTERED_REGION_CODES = ['Kedah']

# Load case file
urlConfirmedOnly = 'data/confirmedOnlyForRt.csv'
statesConfirmedOnly = pd.read_csv(urlConfirmedOnly,
                     usecols=['date', 'state', 'cases'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()

urlConfirmedOnly_dom = 'data/confirmedOnlyForRt_dom.csv'
statesConfirmedOnly_dom = pd.read_csv(urlConfirmedOnly_dom,
                     usecols=['date', 'state', 'cases'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()

urlOnset = 'data/onsetForRt.csv'
statesOnset = pd.read_csv(urlOnset,
                     usecols=['date', 'state', 'cases'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()

urlOnset_dom = 'data/onsetForRt_dom.csv'
statesOnset_dom = pd.read_csv(urlOnset_dom,
                     usecols=['date', 'state', 'cases'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()

revert_to_confirmed_base = False

# Ensure all case diffs are greater than zero
def checkAllDiffPositive(dataToCheck):
    for state, grp in dataToCheck.groupby('state'):
        dataToCheckValues = dataToCheck[state].dropna()
        is_positive = dataToCheckValues.ge(0)
        try:
            assert is_positive.all()
        except AssertionError:
            print(f"Warning: {state} has date with negative case counts")
            print(dataToCheckValues[~is_positive])

checkAllDiffPositive(statesConfirmedOnly)
checkAllDiffPositive(statesConfirmedOnly_dom)
checkAllDiffPositive(statesOnset)
checkAllDiffPositive(statesOnset_dom)

p_EPS = 0.001
#p_onset_comfirmed_delay = pd.read_csv("data/onset_confirmed_delay.csv", index_col=None, header=None, squeeze=True)
P_CONFIRMED_DELAY_T = np.linspace(0, 30, 31)
p_onset_comfirmed_delay_cum = sps.weibull_min(c = 1.741, scale = 8.573).cdf(x = P_CONFIRMED_DELAY_T)
p_onset_comfirmed_delay = p_onset_comfirmed_delay_cum[1:] - p_onset_comfirmed_delay_cum[:-1]
p_onset_comfirmed_delay = np.insert(p_onset_comfirmed_delay, 0, p_EPS)

P_INFECTION_ONSET_DELAY_T = np.linspace(0, 30, 31)
p_infection_onset_delay_cum = sps.lognorm(scale = math.exp(1.519), s = 0.615).cdf(x = P_INFECTION_ONSET_DELAY_T - 0.5)
p_infection_onset_delay = p_infection_onset_delay_cum[1:] - p_infection_onset_delay_cum[:-1]
p_infection_onset_delay = np.insert(p_infection_onset_delay, 0, p_EPS)
p_infection_confirm_delay = np.convolve(p_onset_comfirmed_delay, p_infection_onset_delay)

def backprojNP(confirmed, p_delay, addingExtraRows=False):

    nExtraDays = 0
    if addingExtraRows == True:
        # for stability, append 10 extra days from last date
        nExtraDays = 10
        for i in range(nExtraDays):
            newIndex = [(confirmed.index[-1][0], confirmed.index[-1][1] + timedelta(days=1))]
            #row = pd.Series(index=newIndex, data=confirmed[-1])
            row = pd.Series(index=newIndex, data=[0])
            confirmed = confirmed.append(row)

    # in case of back projection, p_delay means onset to confirmed

    confirmed = confirmed[(confirmed > 0).argmax():]
    # Initial guess - p_delay(onset to confirmed) ~ p_delay(confirmed to onset)
    # Reverse cases so that we convolve into the past

    #convolved = np.convolve(confirmed[::-1], p_delay)[len(p_delay)-1:]
    convolved = np.convolve(confirmed[::-1].values, p_delay)[::-1]
    convolved = convolved[len(convolved) - len(confirmed):]
    # Calculate the new date range and index
    if confirmed.index.name == 'date':
        end_date = confirmed.index[-1]
        dr = pd.date_range(end=end_date, periods=len(convolved))
        index = dr
    else:
        end_date = confirmed.index[-1][confirmed.index.names.index("date")]
        dr = pd.date_range(end=end_date, periods=len(convolved))
        state_name = confirmed.index[-1][0]
        index = pd.MultiIndex.from_product([[state_name], dr], names=['state', 'date'])

    initialGuess = convolved
    maxLoop = 1000
    eps = 0.0001
    onsetRaw = initialGuess
    l = 0
    diff = eps * max(onsetRaw) * 100

    storeOnsetResult = []
    while l < maxLoop and diff > eps * max(onsetRaw):
        # EM step
        mu = np.convolve(onsetRaw, p_delay)
        if len(mu) > len(confirmed):
           mu = mu[:-(len(mu) - len(confirmed))]
        F = p_delay.cumsum()[::-1]
        if len(F) < len(confirmed):
            F = np.pad(F, (len(confirmed) - len(F), 0), 'constant', constant_values = (1, 1))
        sumPart = np.convolve((confirmed / mu)[::-1], p_delay)[::-1][len(p_delay)-1:]
        #if p_delay[0] == 0.0:
        #    sumPart[0] = confirmed[0] / onsetRaw[0]

        if len(sumPart) > len(confirmed):
           sumPart = sumPart[(len(sumPart) - len(confirmed)):]
        onsetEM = onsetRaw / F * sumPart
        #if p_delay[0] == 0.0:
        #    n = len(onsetEM) - 1
        #    onsetEM[n] = onsetRaw[n] * confirmed[n] / mu[n]

        # S step
        k = 2 # k has to be even integer
        w = []
        for i in range(k + 1):
            w.append(spsp.binom(k, i) / (2 ** k))
        intHalfK = (int)(k / 2)
        onsetEMS = np.convolve(w, onsetEM)[intHalfK:-intHalfK]
        diff = max(abs(onsetRaw - onsetEMS))
        l = l + 1
        storeOnsetResult.append(onsetEMS)
        onsetRaw = onsetEMS
    if nExtraDays == 0:
        return pd.Series(onsetEMS, index=index, name=confirmed.name)
    else:
        return pd.Series(onsetEMS[:-nExtraDays], index=index[:-nExtraDays], name=confirmed.name)

def confirmed_to_onset(confirmed, p_onset_comfirmed_delay, backProjection = True):
    if revert_to_confirmed_base:
        return confirmed
    else:
        assert not confirmed.isna().any()
        
        # backprojNP
        if backProjection:
            # in case of back projection, p_onset_comfirmed_delay means onset to confirmed
            onset = backprojNP(confirmed, p_onset_comfirmed_delay, True)
        else:
            # when no back projection, p_onset_comfirmed_delay means confirmed to onset
            # Reverse cases so that we convolve into the past
            convolved = np.convolve(confirmed[::-1].values, p_onset_comfirmed_delay)
            # Calculate the new date range and index
            if confirmed.index.name == 'date':
                end_date = confirmed.index[-1]
                dr = pd.date_range(end=end_date, periods=len(convolved))
                index = dr
            else:
                end_date = confirmed.index[-1][confirmed.index.names.index("date")]
                dr = pd.date_range(end=end_date, periods=len(convolved))
                state_name = confirmed.index[-1][0]
                index = pd.MultiIndex.from_product([[state_name], dr], names=['state', 'date'])

            # Flip the values and assign the date range
            onset = pd.Series(np.flip(convolved), index=index, name=confirmed.name)
        
        return onset

def onset_to_infection(onset, p_infection_onset_delay, backProjection = True):
    if revert_to_confirmed_base:
        return onset
    else:
        assert not onset.isna().any()
        
        # backprojNP
        if backProjection:
            # in case of back projection, p_infection_onset_delay means onset to confirmed
            onset = backprojNP(onset, p_infection_onset_delay, True)
        else:
            # Reverse cases so that we convolve into the past
            convolved = np.convolve(onset[::-1].values, p_infection_onset_delay)
        
            # Calculate the new date range and index
            if onset.index.name == 'date':
                end_date = onset.index[-1]
                dr = pd.date_range(end=end_date, periods=len(convolved))
                index = dr
            else:
                end_date = onset.index[-1][onset.index.names.index("date")]
                dr = pd.date_range(end=end_date, periods=len(convolved))
                state_name = onset.index[-1][0]
                index = pd.MultiIndex.from_product([[state_name], dr], names=['state', 'date'])
        
            # Flip the values and assign the date range
            onset = pd.Series(np.flip(convolved), index=index, name=onset.name)
        
        return onset


def adjust_onset_for_right_censorship(onset, p_onset_comfirmed_delay):
    if revert_to_confirmed_base:
        return onset, 0
    else:
        cumulative_p_delay = p_onset_comfirmed_delay.cumsum()
        
        # Calculate the additional ones needed so shapes match
        ones_needed = len(onset) - len(cumulative_p_delay)
        padding_shape = (0, ones_needed)
        
        # Add ones and flip back
        cumulative_p_delay = np.pad(
            cumulative_p_delay,
            padding_shape,
            constant_values=1)
        cumulative_p_delay = np.flip(cumulative_p_delay)
        
        # Adjusts observed onset values to expected terminal onset values
        adjusted = onset / cumulative_p_delay
        
        return adjusted, cumulative_p_delay

def adjust_infection_for_right_censorship(infection, p_infection_onset_delay):
    if revert_to_confirmed_base:
        return onset, 0
    else:
        cumulative_p_delay = p_infection_onset_delay.cumsum()
        
        # Calculate the additional ones needed so shapes match
        ones_needed = len(infection) - len(cumulative_p_delay)
        padding_shape = (0, ones_needed)
        
        # Add ones and flip back
        cumulative_p_delay = np.pad(
            cumulative_p_delay,
            padding_shape,
            constant_values=1)
        cumulative_p_delay = np.flip(cumulative_p_delay)
        
        # Adjusts observed onset values to expected terminal onset values
        adjusted = infection / cumulative_p_delay
        
        # not doing right-side adjustment now
        return infection, cumulative_p_delay
        #return adjusted, cumulative_p_infection_onset_delay

def prepare_cases_old(new_cases, cutoff=25):
    #new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    idx_start = np.argmax(smoothed >= cutoff)
    #idx_start = np.searchsorted(smoothed, cutoff)
    
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]
    
    return original, smoothed

def prepare_cases(new_cases, cutoff=25):

    #new_cases = cases.diff()
    #idx_start = np.searchsorted(new_cases, cutoff)
    #new_cases = new_cases.iloc[idx_start:]
    #truncated = new_cases.loc[new_cases.index]
    idx_start = np.argmax(new_cases >= cutoff)
    if idx_start == 0 and new_cases[0] < cutoff:
        return []
#    idx_start = np.searchsorted(new_cases, cutoff)
    new_cases = new_cases.iloc[idx_start:]
    truncated = new_cases.loc[new_cases.index]
    if truncated.index.names == [None]: # only if time series has single index
        truncated.index.rename('date', inplace = True)

    return truncated

#if revert_to_confirmed_base:
#    original, adjusted = prepare_cases_old(adjusted, cutoff=10)
#else:
#    adjusted = prepare_cases(adjusted, cutoff=2)



# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
# https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
#GAMMA = 1/7

#Function for Calculating the Posteriors
def get_posteriors(sr, sr_dom, sigma=0.15):
    ## (0) Round adjusted cases for poisson distribution
    #sr = sr.round()

    expectSerialIntervalOverLikelihood = False

    likelihoods = np.full_like(sr[:-1].values - r_t_range[:, None], 0.0)
    serial_interval_cumdensity = 0.
    prevTau = 0.
    if expectSerialIntervalOverLikelihood:
        taus = np.linspace(0.1, 10.0, 20)
        #taus = [7.]
        for tau in taus:
            #serial_interval_density = sps.gamma.pdf(a = 6.0, scale = 1./1.5,
            #x=tau)
            serial_interval_density = sps.weibull_min.cdf(c=2.305, scale=5.452, x=tau) - sps.weibull_min.cdf(c=2.305, scale=5.452, x=prevTau)
            GAMMA = 1. / tau

            # (1) Calculate Lambda
            lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1.))
        
            # (2) Calculate each day's likelihood
            likelihood = pd.DataFrame(data = sps.gamma(a=sr_dom[1:] + 1., scale=1., loc=0.).pdf(x=lam),
                #data = sps.poisson.pmf(sr[1:].values, lam),
                index = r_t_range,
                columns = sr.index[1:])
            likelihoods += likelihood * serial_interval_density
            serial_interval_cumdensity += serial_interval_density
            prevTau = tau

        likelihoods /= serial_interval_cumdensity
    else:
        lam = np.full_like(sr[:-1].values - r_t_range[:, None], 0.0)
        taus = np.linspace(0., 60.0, 61)
        #taus = [7.]
        sameAsNishiuraSum = True

        if sameAsNishiuraSum:
            for i in range(1, min((int)(taus[-1])+1, len(sr))):
                tau = taus[i]
                serial_interval_density = sps.weibull_min.cdf(c=2.305, scale=5.452, x=tau) - sps.weibull_min.cdf(c=2.305, scale=5.452, x=prevTau)
                #GAMMA = 1./tau
    
                # (1) Calculate Lambda
                paddedSr = np.pad(sr[:-i].values, (i - 1, 0), 'constant', constant_values = (0., 0.))
                lam += r_t_range[:, None] * paddedSr * serial_interval_density
                serial_interval_cumdensity += serial_interval_density
                prevTau = tau

            lam /= serial_interval_cumdensity
        else:
            for i in range(1, min((int)(taus[-1])+1, len(sr))):
                tau = taus[i]
                #serial_interval_density = sps.gamma.pdf(a = 6.0, scale = 1./1.5, x=tau)
                serial_interval_density = sps.weibull_min.cdf(c=2.305, scale=5.452, x=tau) - sps.weibull_min.cdf(c=2.305, scale=5.452, x=prevTau)
                #GAMMA = 1./tau
    
                # (1) Calculate Lambda
                paddedSr = np.pad(sr[:-i].values, (i - 1, 0), 'constant', constant_values = (0., 0.))
                lam += paddedSr * np.exp(r_t_range[:, None] - 1.) * serial_interval_density
                serial_interval_cumdensity += serial_interval_density
                prevTau = tau

            lam /= serial_interval_cumdensity

        # (2) Calculate each day's likelihood
        likelihoods = pd.DataFrame(data = sps.gamma(a=sr_dom[1:] + 1., scale=1., loc=0.).pdf(x=lam),
            #data = sps.poisson.pmf(sr[1:].values, lam),
            index = r_t_range,
            columns = sr.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                             ).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)
    
    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    makePosterior = True
    if makePosterior:
        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

            #(5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]
            
            #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior
            
            #when sr[i-1] = 0 (then lam(sr[i-1], r_t_range) = 0, so denominator = 0
            #Then we take limit of sr[i] -> +0
            if sr[previous_day] == 0.0:
                numerator = 1. / factorial(sr_dom[current_day]) * np.exp(GAMMA * (r_t_range - 1.) * sr_dom[current_day]) * current_prior

            #(5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            posteriors[current_day] = numerator/denominator
        
            # Add to the running sum of log likelihoods
            log_likelihood += np.log(denominator)
    else:
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
            numerator = likelihoods[current_day]
            denominator = np.sum(numerator)
            posteriors[current_day] = numerator / denominator
            log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

## Note that we're fixing sigma to a value just for the example
#posteriors, log_likelihood = get_posteriors(adjusted, sigma=.25)


##Result
#ax = posteriors.plot(title=f'{state_name} - Daily Posterior for $R_t$',
#           legend=False, 
#           lw=1,
#           c='k',
#           alpha=.3,
#           xlim=(0.4,6))

#ax.set_xlabel('$R_t$');
##plt.show()

def highest_density_interval(pmf, p=.9, debug=False):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    
    if cumsum[0] > 1 - p:
        low = pmf.index[0]
        highIdx = (cumsum < p).argmin()
        high = pmf.index[highIdx]
    else:
        # N x N matrix of total probability mass for each low, high
        total_p = cumsum - cumsum[:, None]
    
        # Return all indices with total_p > p
        lows, highs = (total_p > p).nonzero()
        
        # Find the smallest range (highest density)
        best = (highs - lows).argmin()
        
        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]
        
    return pd.Series([low, high],
                     index=[f'Low_{p*100:.0f}',
                            f'High_{p*100:.0f}'])


## Note that this takes a while to execute - it's not the most efficient algorithm
#hdis = highest_density_interval(posteriors, p=.9)

#most_likely = posteriors.idxmax().rename('ML')

## Look into why you shift -1
#result = pd.concat([most_likely, hdis], axis=1)

#result.tail()

#def plot_rt(result, ax, state_name):
    
#    ax.set_title(f"{state_name}")
    
#    # Colors
#    ABOVE = [1,0,0]
#    MIDDLE = [1,1,1]
#    BELOW = [0,0,0]
#    cmap = ListedColormap(np.r_[
#        np.linspace(BELOW,MIDDLE,25),
#        np.linspace(MIDDLE,ABOVE,25)
#    ])
#    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
#    index = result['ML'].index.get_level_values('date')
#    values = result['ML'].values
    
#    # Plot dots and line
#    ax.plot(index, values, c='k', zorder=1, alpha=.25)
#    ax.scatter(index,
#               values,
#               s=40,
#               lw=.5,
#               c=cmap(color_mapped(values)),
#               edgecolors='k', zorder=2)
    
#    # Aesthetically, extrapolate credible interval by 1 day either side
#    lowfn = interp1d(date2num(index),
#                     result['Low_90'].values,
#                     bounds_error=False,
#                     fill_value='extrapolate')
    
#    highfn = interp1d(date2num(index),
#                      result['High_90'].values,
#                      bounds_error=False,
#                      fill_value='extrapolate')
    
#    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
#                             end=index[-1]+pd.Timedelta(days=1))
    
#    ax.fill_between(extended,
#                    lowfn(date2num(extended)),
#                    highfn(date2num(extended)),
#                    color='k',
#                    alpha=.1,
#                    lw=0,
#                    zorder=3)

#    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
#    # Formatting
#    ax.xaxis.set_major_locator(mdates.MonthLocator())
#    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
#    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
#    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
#    ax.yaxis.tick_right()
#    ax.spines['left'].set_visible(False)
#    ax.spines['bottom'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.margins(0)
#    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
#    ax.margins(0)
#    ax.set_ylim(0.0, 5.0)
#    ax.set_xlim(pd.Timestamp('2020-02-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
#    fig.set_facecolor('w')

    
#fig, ax = plt.subplots(figsize=(600/72,400/72))

#plot_rt(result, ax, state_name)
#ax.set_title(f'Real-time $R_t$ for {state_name}')
#ax.xaxis.set_major_locator(mdates.WeekdayLocator())
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

##plt.show()

#Choose an optimum sigma
sigmas = np.linspace(0.01, 1.0, 20)
#sigmas = []
#sigmas.append(3.0)

targets = ~statesConfirmedOnly.index.get_level_values('state').isin(FILTERED_REGION_CODES)
states_to_process = statesConfirmedOnly.loc[targets]

results = {}
adjustedCases = {}
adjustedCases_dom = {}

for state_name, cases in states_to_process.groupby(level='state'):
    #if state_name != 'Johor':
    #if state_name != 'AK':
    #    continue

    print(state_name)
    #cases = cases.diff().dropna()
    onsetFromConfirmed = confirmed_to_onset(cases, p_onset_comfirmed_delay)
    onset = statesOnset.filter(like=state_name, axis=0)
    onsetOriginal = onset
    onset = onset + onsetFromConfirmed
    onset = onset.dropna()
#    adjustedOnset, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_onset_comfirmed_delay)
    adjustedOnset = onset
    infected = onset_to_infection(adjustedOnset, p_infection_onset_delay)
    #adjusted = infected
    adjusted, cumulative_p_infection_onset_delay = adjust_infection_for_right_censorship(infected, p_infection_confirm_delay)

    onset_dom = statesOnset_dom.filter(like=state_name, axis=0)
    onsetOriginal_dom = onset_dom
    onsetFromConfirmed_dom = confirmed_to_onset(
        statesConfirmedOnly_dom.filter(like=state_name, axis=0), p_onset_comfirmed_delay)
    #onset_dom = onset_dom + onsetFromConfirmed_dom
    onset_dom = onset_dom + onsetFromConfirmed_dom
    onset_dom = onset_dom.dropna()
    adjustedOnset_dom = onset_dom
    #adjustedOnset_dom, cumulative_p_delay = adjust_onset_for_right_censorship(onset_dom, p_infection_confirm_delay)
    infected_dom = onset_to_infection(adjustedOnset_dom, p_infection_onset_delay)
    #adjusted_dom = infected_dom
    adjusted_dom, cumulative_p_infection_onset_delay = adjust_infection_for_right_censorship(infected_dom, p_infection_confirm_delay)

    #onsetOriginal.to_csv(state_name + "_onsetOriginal.csv")
    #onset.to_csv(state_name + "_onset.csv")
    #infected.to_csv(state_name + "_infected.csv")
    #adjusted.to_csv(state_name + "_adjusted.csv")
    #onsetOriginal_dom.to_csv(state_name + "_onsetOriginal_dom.csv")
    #onsetFromConfirmed_dom.to_csv(state_name + "_onsetFromConfirmed_dom.csv")
    #onset_dom.to_csv(state_name + "_onset_dom.csv")
    #infected_dom.to_csv(state_name + "_infected_dom.csv")
    #adjusted_dom.to_csv(state_name + "_adjusted_dom.csv")

    #if revert_to_confirmed_base:
    #    original, trimmed = prepare_cases_old(adjusted, cutoff=10)
    #else:
    #    trimmed = prepare_cases(adjusted, cutoff=10)
    
    #if len(trimmed) == 0:
    #    if revert_to_confirmed_base:
    #        original, trimmed = prepare_cases_old(adjusted, cutoff=2)
    #    else:
    #        trimmed = prepare_cases(adjusted, cutoff=2)
    #    if len(trimmed) == 0:
    #        print(state_name + ": too few cases for statistics")
    #        continue

    if revert_to_confirmed_base:
        original, trimmed = prepare_cases_old(adjusted, cutoff=2)
        trimmed_dom = adjusted_dom[trimmed.index]
    else:
        trimmed = prepare_cases(adjusted, cutoff=2)
        trimmed_dom = adjusted_dom[trimmed.index]
    if len(trimmed) == 0 or len(trimmed_dom) == 0:
        print(state_name + ": too few cases for statistics")
        continue
    adjusted = trimmed
    adjustedCases[state_name] = adjusted
    adjusted_dom = trimmed_dom
    adjustedCases_dom[state_name] = adjusted_dom

    # Include uncertainty of adjustment on onset on right side because we don't know future confirmed cases to compute recent onset cases
    # t: target date
    # T: latest date
    # L: delay limit, min(L; cumulative(p_infection_onset_delay(0 to L)) > threshould)
    P_DELAY_THRESHOLD = 0.99
    L = np.argmax(np.cumsum(p_infection_onset_delay) > P_DELAY_THRESHOLD)
    no_need_adjust = adjusted[:-L].values
    stdCases = []
    if len(no_need_adjust) >= L:
        for lag in range(1, L):
            stdCases.append(np.std(no_need_adjust[lag:] - no_need_adjust[:-lag]))

    result = {}

    # store std from uncertainty of future confirmed cases for later adjustment
    result['std_future_cases'] = stdCases
    #result['std_future_rt'] = stds_case_ratios

    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    
    for sigma in sigmas:
        posteriors, log_likelihood = get_posteriors(adjusted, adjusted_dom, sigma=sigma)
        #posteriors, log_likelihood = get_posteriors(adjusted, adjusted_dom, sigma=sigma)
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    
    # Store all results keyed off of state name
    results[state_name] = result
    #clear_output(wait=True)

print('Done.')

# Each index of this array holds the total of the log likelihoods for
# the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for state_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

# Select the value that has the highest log likelihood
sigma = sigmas[max_likelihood_index]
print("optimum sigma: " + str(sigma))

## Plot it
#fig, ax = plt.subplots()
#ax.set_title(f"Maximum Likelihood value for $\sigma$ = {sigma:.2f}");
#ax.plot(sigmas, total_log_likelihoods)
#ax.axvline(sigma, color='k', linestyle=":")

def posteriors_get_max_point(posteriors):
    max_rts = posteriors.idxmax().rename('ML')
    indexes = posteriors.index
    for t in range(len(max_rts)):
        posterior = posteriors.iloc[:, t]
        max_rt = max_rts[t]
        max_index = indexes.get_loc(max_rt)
        max_post = posterior.iloc[max_index]

        if(max_index == 0 or max_index == len(indexes) - 1):
            max_rts[t] = max_rt
            continue

        max_rt_m1 = indexes[max_index - 1]
        max_post_m1 = posterior.iloc[max_index - 1]
        max_rt_p1 = indexes[max_index + 1]
        max_post_p1 = posterior.iloc[max_index + 1]

        if max_index - 2 < 0 or max_index + 2 >= len(indexes):
            max_rts[t] = max_rt
            continue

        if max_post_m1 < max_post_p1: # true maximum is between max_rt and max_rt_p1
             max_rt_p2 = indexes[max_index + 2]
             max_post_p2 = posterior.iloc[max_index + 2]
             d1 = (max_post - max_post_m1) / (max_rt - max_rt_m1)
             s1 = - d1 * max_rt + max_post
             d2 = (max_post_p2 - max_post_p1) / (max_rt_p2 - max_rt_p1)
             s2 = - d2 * max_rt_p2 + max_post_p2
 
        else: # true maximum is between max_rt and max_rt_m1
            max_rt_m2 = indexes[max_index - 2]
            max_post_m2 = posterior.iloc[max_index - 2]
            d1 = (max_post - max_post_p1) / (max_rt - max_rt_p1)
            s1 = - d1 * max_rt + max_post
            d2 = (max_post_m2 - max_post_m1) / (max_rt_m2 - max_rt_m1)
            s2 = - d2 * max_rt_m2 + max_post_m2
        rt = (s2 - s1)/(d1 - d2)
        max_rts[t] = rt
    return max_rts

# modify posteriors to include the uncertainty due to future confirmed cases
for state_name, result in results.items():
    posteriors = result['posteriors'][max_likelihood_index]
    most_likely = posteriors_get_max_point(posteriors)
    #stds_future_rt = result['std_future_rt']

    # simulate Rt with bump on future confirmed case
    L = np.argmax(np.cumsum(p_infection_onset_delay) > P_DELAY_THRESHOLD)
    most_likely_bumpeds = []
    adjusted0 = adjustedCases[state_name]
    # if series is not long enough to compute stdCases (thus Null),
    # the series is too short and not statistically meaningful to calculate stdCases
    # So we do not make adjustment for uncertaity of future cases modulation
    # This no-treating strategy should be alright because confidence interval from Poisson/gamma distribution is already large
    stdCases = result['std_future_cases']
    #if stdCases:
    if 1 == 2:
        bump_size = max(stdCases)
        for d in range(1, L):
            if len(adjusted0) - len(p_infection_onset_delay) + d > 0:
                p_delay_values = np.concatenate((p_infection_onset_delay.values[d:], np.zeros(len(adjusted0) - len  (p_infection_onset_delay) + d)))
            else:
                p_delay_values = np.resize(p_infection_onset_delay.values[d:], (1, len(adjusted0)))[0]
            bumpedAdjusted = adjusted0 + bump_size * p_delay_values[::-1] # +1 of the confirmed cases of d-day later
            bumpedPosteriors, dummy = get_posteriors(bumpedAdjusted, sigma=sigma)
            most_likely_bumped = posteriors_get_max_point(bumpedPosteriors)
            most_likely_change = (most_likely_bumped - most_likely) / bump_size
            most_likely_bumpeds.append(most_likely_change)
    
        stdRts = 0. * most_likely
        for d in range(1, L):
            stdRts += most_likely_bumpeds[d-1]**2 * stdCases[d-1]**2
        stdRts = np.sqrt(stdRts)[::-1]
    #    stdRts.to_csv("stdRts.csv")
    
        # modify posteriors
        for d in range(len(stdCases)):
            if stdRts[d] < 0.001:
                continue
            normal_dist = sps.norm(loc=r_t_range, scale=stdRts[d]).pdf(r_t_range[:,None])
            normal_dist /= normal_dist.sum(axis=0)
            normal_dist /= normal_dist.sum(axis=1)[:,None]
            posterior_to_be_modified = posteriors.iloc[:,-d-1]
            posterior_modified = posterior_to_be_modified @ normal_dist
            posteriors.iloc[:,-d-1] = posterior_modified
    
            #posteriorsfilename_before = posterior_to_be_modified.name[0] + posterior_to_be_modified.name   [1].strftime('%Y-%m-%d') + ".csv"
            #posterior_to_be_modified.to_csv(posteriorsfilename_before)
            #posteriorsfilename_after = posterior_to_be_modified.name[0] + posterior_to_be_modified.name    [1].strftime('%Y-%m-%d') + "_after.csv"
            #np.savetxt(posteriorsfilename_after, posterior_modified, delimiter=",")
            #normalfilename = posterior_to_be_modified.name[0] + posterior_to_be_modified.name[1].strftime  ('%Y-%m-%d') + "_normal_dist.csv"
            #np.savetxt(normalfilename, normal_dist, delimiter=",")

    result['posteriors'][max_likelihood_index] = posteriors 


#Compile Final Results
final_results = None

for state_name, result in results.items():
    print(state_name)
    posteriors = result['posteriors'][max_likelihood_index]
    hdis_90 = highest_density_interval(posteriors, p=.9)
    hdis_50 = highest_density_interval(posteriors, p=.5)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis_90, hdis_50], axis=1)
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])

print('Done final result.')

#Save final result to csv files
for state_name, final_result in final_results.groupby("state"):
    final_result.loc[state_name].to_csv("data/" + state_name + ".csv")
print('Saved final result.')



