import sys
print(sys.version)
# For some reason Theano is unhappy when I run the GP, need to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import os.path
import requests
import pandas as pd
import numpy as np
import scipy.stats as sps
from scipy.special import factorial
import math
import scipy.special as spsp

from datetime import date
from datetime import datetime
from datetime import timedelta

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

#Choose an optimum sigma
sigmas = np.linspace(0.01, 1.0, 20)
#sigmas = []
#sigmas.append(3.0)

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

def backprojNP(confirmed, p_delay, addingExtraRows=False, extraRowsWithLastValue=True):

    # To avoid 0.0/0.0 during computation, we add a tiny value to input data, and subtract the tiny value from and floor the result by 0.
    epsilon = 0.001
    confirmed = confirmed + epsilon

    nExtraDays = 0
    if addingExtraRows == True:
        # for stability, append 20 extra days from last date
        nExtraDays = 20
        for i in range(nExtraDays):
            newIndex = [(confirmed.index[-1][0], confirmed.index[-1][1] + timedelta(days=1))]
            if extraRowsWithLastValue:
                row = pd.Series(index=newIndex, data=confirmed[-1])
            else:
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
        #k = 30 # k has to be even integer
        k = 2 * (int)(p_delay.argmax(0)**2 / 2) # k has to be even integer
        w = []
        for i in range(k + 1):
            w.append(spsp.binom(k, i) / (2 ** k))
        intHalfK = (int)(k / 2)
        onsetEMS = np.convolve(w, onsetEM)[intHalfK:-intHalfK]
        diff = max(abs(onsetRaw - onsetEMS))
        l = l + 1
        storeOnsetResult.append(onsetEMS)
        onsetRaw = onsetEMS

    # Because we added a tiny value to the input data, we subtract the tiny value and floor the result data by 0
    onsetEMS = onsetEMS - epsilon
    onsetEMS[onsetEMS < 0.0] = 0.0

    if nExtraDays == 0:
        return pd.Series(onsetEMS, index=index, name=confirmed.name)
    else:
        return pd.Series(onsetEMS[:-nExtraDays], index=index[:-nExtraDays], name=confirmed.name)

def confirmed_to_onset(confirmed, p_onset_comfirmed_delay, revert_to_confirmed_base, rightCensorshipByDelayFunctionDevision, backProjection):
    if revert_to_confirmed_base:
        return confirmed
    else:
        assert not confirmed.isna().any()
        
        # backprojNP
        if backProjection:
            # in case of back projection, p_onset_comfirmed_delay means onset to confirmed
            onset = backprojNP(confirmed, p_onset_comfirmed_delay, True, not(rightCensorshipByDelayFunctionDevision))
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

def onset_to_infection(onset, p_infection_onset_delay, revert_to_confirmed_base, rightCensorshipByDelayFunctionDevision, backProjection):
    if revert_to_confirmed_base:
        return onset
    else:
        assert not onset.isna().any()
        
        # backprojNP
        if backProjection:
            # in case of back projection, p_infection_onset_delay means onset to confirmed
            onset = backprojNP(onset, p_infection_onset_delay, True, not(rightCensorshipByDelayFunctionDevision))
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


def adjust_onset_for_right_censorship(onset, p_onset_comfirmed_delay, revert_to_confirmed_base, obsDate):
    if revert_to_confirmed_base:
        return onset, 0
    else:
        cumulative_p_delay = p_onset_comfirmed_delay.cumsum()

        lastDate = onset.index[-1][1]
        delayOffsetDays = (obsDate - lastDate).days
        cumulative_p_delay = cumulative_p_delay[delayOffsetDays:]
        
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

def adjust_infection_for_right_censorship(infection, p_infection_onset_delay, revert_to_confirmed_base, obsDate):
    if revert_to_confirmed_base:
        return infection, 0
    else:
        cumulative_p_delay = p_infection_onset_delay.cumsum()

        lastDate = infection.index[-1][1]
        delayOffsetDays = (obsDate - lastDate).days
        cumulative_p_delay = cumulative_p_delay[delayOffsetDays:]

        
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
        #return infection, cumulative_p_delay
        return adjusted, cumulative_p_delay

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

    idx_start = np.argmax(new_cases >= cutoff)
    if idx_start == 0 and new_cases[0] < cutoff:
        return []
    new_cases = new_cases.iloc[idx_start:]
    truncated = new_cases.loc[new_cases.index]
    if truncated.index.names == [None]: # only if time series has single index
        truncated.index.rename('date', inplace = True)

    return truncated


#Function for Calculating the Posteriors
def get_posteriors(sr, sr_dom, singleTau, sumStyle, includePosterior, sigma=0.15):
    serial_interval_cumdensity = 0.
    prevTau = 0.
    #lam = np.full_like(sr[:-1].values - r_t_range[:, None], 0.0)
    lam = pd.DataFrame(data = 0.,
        index = r_t_range,
        columns = sr.index[1:])

    if singleTau:
        # Gamma is 1/serial interval
        # https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
        # https://www.nejm.org/doi/full/10.1056/NEJMoa2001316
        taus = [7.]
    else:
        taus = np.linspace(0., 60.0, 61)

    for i in range(1, min((int)(taus[-1])+1, len(sr))):
        tau = taus[i]
        #serial_interval_density = sps.gamma.pdf(a = 6.0, scale = 1./1.5, x=tau)
        serial_interval_density = sps.weibull_min.cdf(c=2.305, scale=5.452, x=tau) - sps.weibull_min.cdf(c=2.305, scale=5.452, x=prevTau)

        # (1) Calculate Lambda
        if sumStyle == "Nishiura":
            paddedSr = np.pad(sr[:-i].values, (i - 1, 0), 'constant', constant_values = (0., 0.))
            lam += r_t_range[:, None] * paddedSr * serial_interval_density
        elif sumStyle == "Exponential":
            paddedSr = np.pad(sr[:-i].values, (i - 1, 0), 'constant', constant_values = (0., 0.))
            lam += paddedSr * np.exp(r_t_range[:, None] - 1.) * serial_interval_density
        elif sumStyle == "K-Sys":
            GAMMA = 1. / tau
            lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1.))

        serial_interval_cumdensity += serial_interval_density
        prevTau = tau
    lam /= serial_interval_cumdensity

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(data = sps.gamma(a=sr_dom[1:] + 1., scale=1., loc=0.).pdf(x=lam),
        #data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])

    ##when sr[i-1] = 0 (then lam(sr[i-1], r_t_range) = 0, so denominator = 0
    ##Then we take limit of sr[i-1] -> +0
    #if sr[previous_day] == 0.0:
    #    numerator = 1. / factorial(sr_dom[current_day]) * np.exp(GAMMA * (r_t_range - 1.) *sr_do[current_day]) * current_prior

    # When daily new case continues to be 0, likelihoods[current_day] becomes zero for all Rt.
    # It means Rt is undetermined by recent cases.
    # Only constraint current_prior which contains Gaussian(Rt - Rt[previous_day])
    # To achieve this, we fill likelihoods[current_day] with 1.0 for all Rt
    for i in range(len(likelihoods.columns)):
        likelihoods_i = likelihoods[likelihoods.columns[i]]
        if len(likelihoods_i[likelihoods_i != 0.0]) == 0:
            likelihoods[likelihoods.columns[i]].values = [1.0] * len(likelihoods_i)

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

    if includePosterior:
        # (5) Iteratively apply Bayes' rule
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

            #(5a) Calculate the new prior
            current_prior = process_matrix @ posteriors[previous_day]
            
            #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
            numerator = likelihoods[current_day] * current_prior
            
            #(5c) Calcluate the denominator of Bayes' Rule P(k)
            denominator = np.sum(numerator)

            # Execute full Bayes' Rule
            posteriors[current_day] = numerator/denominator
        
            # Add to the running sum of log likelihoods
            if denominator == 0.0:
                print("zero")
            log_likelihood += np.log(denominator)
    else:
        for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
            numerator = likelihoods[current_day]
            denominator = np.sum(numerator)
            posteriors[current_day] = numerator / denominator
            log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

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

def diffCumulativeCase(cumulativeCase):
    if type(cumulativeCase) is list and cumulativeCase == []:
        return []
    else:
        diffCase = cumulativeCase[cumulativeCase != cumulativeCase] # no rows but only copy index and column
        for state, grp in cumulativeCase.groupby('state'):
            diffCase = diffCase.append(grp.diff().dropna())
        return diffCase

def computeRt(statesOnset, statesOnset_dom, statesConfirmedOnly, statesConfirmedOnly_dom, p_onset_comfirmed_delay, p_infection_onset_delay, p_infection_confirm_delay, isCaseCumulative, includePosterior = True, sumStyle = "Nishiura", rightCensorshipByDelayFunctionDevision = False, singleTau = False, obsDate = None, FILTERED_REGION_CODES = [''], revert_to_confirmed_base = False, backProjection=True):
    if isCaseCumulative:
        statesOnset = diffCumulativeCase(statesOnset)
        statesOnset_dom = diffCumulativeCase(statesOnset_dom)
        statesConfirmedOnly = diffCumulativeCase(statesConfirmedOnly)
        statesConfirmedOnly_dom = diffCumulativeCase(statesConfirmedOnly_dom)

    onsetHasContent = type(statesOnset) is not list or statesOnset != []
    onset_domHasContent = type(statesOnset_dom) is not list or statesOnset_dom != []

    if onsetHasContent: checkAllDiffPositive(statesOnset)
    if onset_domHasContent: checkAllDiffPositive(statesOnset_dom)
    checkAllDiffPositive(statesConfirmedOnly)
    checkAllDiffPositive(statesConfirmedOnly_dom)
    
    if obsDate == None:
        lastDateConfirmedOnly = max(statesConfirmedOnly.index.get_level_values('date'))
        lastDateConfirmedOnly_dom = max(statesConfirmedOnly_dom.index.get_level_values('date'))
        lastDateOnset = max(statesOnset.index.get_level_values('date')) if onsetHasContent else lastDateConfirmedOnly
        lastDateOnset_dom = max(statesOnset_dom.index.get_level_values('date')) if onset_domHasContent else lastDateConfirmedOnly_dom
        obsDate = max(lastDateConfirmedOnly, lastDateConfirmedOnly_dom, lastDateOnset, lastDateOnset_dom)
        #lastDateOnsetUnix = statesOnset.index[-1][1].value / 10**9 if onsetHasContent else 0
        #lastDateOnsetUnix_dom = statesOnset_dom.index[-1][1].value / 10**9 if onset_domHasContent else 0
        #lateDateConfirmedOnly = statesConfirmedOnly.index[-1][1].value / 10**9
        #lateDateConfirmedOnly_dom = statesConfirmedOnly_dom.index[-1][1].value / 10**9
        #obsDate = pd.Timestamp(max(lastDateOnsetUnix, lastDateOnsetUnix_dom, lateDateConfirmedOnly, lateDateConfirmedOnly_dom), unit='s')
    #targets = ~statesConfirmedOnly.index.get_level_values('state').isin(FILTERED_REGION_CODES)
    targets = ~statesConfirmedOnly.index.get_level_values('state').isin(FILTERED_REGION_CODES)
    statesConfirmedlOnly_to_process = statesConfirmedOnly.loc[targets]
    
    results = {}
    adjustedCases = {}
    adjustedCases_dom = {}
    
    for state_name, confirmedOnly in statesConfirmedlOnly_to_process.groupby(level='state'):
        #if state_name != 'Johor':
        #if state_name != 'AK':
        #    continue
    
        print(state_name)
        onsetFromConfirmedOnly = confirmed_to_onset(confirmed=confirmedOnly, p_onset_comfirmed_delay=p_onset_comfirmed_delay, revert_to_confirmed_base=revert_to_confirmed_base, rightCensorshipByDelayFunctionDevision=rightCensorshipByDelayFunctionDevision, backProjection=backProjection)
        if rightCensorshipByDelayFunctionDevision:
            adjustedOnsetConfirmedOnly, cumulative_p_delay = adjust_onset_for_right_censorship(onsetFromConfirmedOnly,  p_onset_comfirmed_delay, revert_to_confirmed_base, obsDate)
        else:
            adjustedOnsetConfirmedOnly = onsetFromConfirmedOnly

        if onsetHasContent:
            onset = statesOnset.filter(like=state_name, axis=0)
            onsetOriginal = onset
            adjustedOnset = onset + adjustedOnsetConfirmedOnly
        else:
            adjustedOnset = adjustedOnsetConfirmedOnly
        adjustedOnset = adjustedOnset.dropna()

        infected = onset_to_infection(onset=adjustedOnset, p_infection_onset_delay=p_infection_onset_delay, revert_to_confirmed_base=revert_to_confirmed_base, rightCensorshipByDelayFunctionDevision=rightCensorshipByDelayFunctionDevision, backProjection=backProjection)
        if rightCensorshipByDelayFunctionDevision:
            adjusted, cumulative_p_infection_onset_delay = adjust_infection_for_right_censorship(infected,  p_infection_confirm_delay, revert_to_confirmed_base, obsDate)
        else:
            adjusted = infected
        confirmedOnly_dom = statesConfirmedOnly_dom.filter(like=state_name, axis=0)
        onsetFromConfirmedOnly_dom = confirmed_to_onset(
            confirmed=confirmedOnly_dom, p_onset_comfirmed_delay=p_onset_comfirmed_delay, revert_to_confirmed_base=revert_to_confirmed_base, rightCensorshipByDelayFunctionDevision=rightCensorshipByDelayFunctionDevision, backProjection=backProjection)
        if rightCensorshipByDelayFunctionDevision:
            adjustedOnsetConfirmedOnly_dom, cumulative_p_delay = adjust_onset_for_right_censorship  (onsetFromConfirmedOnly_dom, p_infection_confirm_delay)
        else:
            adjustedOnsetConfirmedOnly_dom = onsetFromConfirmedOnly_dom
        if onset_domHasContent:
            onset_dom = statesOnset_dom.filter(like=state_name, axis=0)
            onsetOriginal_dom = onset_dom
            adjustedOnset_dom = onset_dom + adjustedOnsetConfirmedOnly_dom
        else:
            adjustedOnset_dom = adjustedOnsetConfirmedOnly_dom
        adjustedOnset_dom = adjustedOnset_dom.dropna()

        infected_dom = onset_to_infection(onset=adjustedOnset_dom, p_infection_onset_delay=p_infection_onset_delay, revert_to_confirmed_base=revert_to_confirmed_base, rightCensorshipByDelayFunctionDevision=rightCensorshipByDelayFunctionDevision, backProjection=backProjection)
        if rightCensorshipByDelayFunctionDevision:
            adjusted_dom, cumulative_p_infection_onset_delay = adjust_infection_for_right_censorship(infected_dom,  p_infection_confirm_delay, revert_to_confirmed_base, obsDate)
        else:
            adjusted_dom = infected_dom
    
        #onsetOriginal.to_csv(state_name + "_onsetOriginal.csv")
        #onset.to_csv(state_name + "_onset.csv")
        #infected.to_csv(state_name + "_infected.csv")
        #adjusted.to_csv(state_name + "_adjusted.csv")
        #onsetOriginal_dom.to_csv(state_name + "_onsetOriginal_dom.csv")
        #confirmedOnly.to_csv(state_name + "_confirmedOnly.csv")
        #onsetFromConfirmedOnly.to_csv(state_name + "_onsetFromConfirmed.csv")
        #adjustedOnsetConfirmedOnly.to_csv(state_name + "_adjustedOnsetConfirmedOnly.csv")
        #adjusted.to_csv(state_name + "_adjusted.csv")
        #confirmedOnly_dom.to_csv(state_name + "_confirmedOnly_dom.csv")
        #onsetFromConfirmedOnly_dom.to_csv(state_name + "_onsetFromConfirmed_dom.csv")
        #onset_dom.to_csv(state_name + "_onset_dom.csv")
        #infected_dom.to_csv(state_name + "_infected_dom.csv")
        #adjusted_dom.to_csv(state_name + "_adjusted_dom.csv")
    
        if revert_to_confirmed_base:
            original, trimmed = prepare_cases_old(adjusted, cutoff=2)
        else:
            trimmed = prepare_cases(adjusted, cutoff=2)

        if len(trimmed) == 0:
            print(state_name + ": too few cases for statistics")
            continue
        else:
            trimmed_dom = adjusted_dom[trimmed.index]
            if len(trimmed_dom) == 0:
                print(state_name + ": too few domestic cases for statistics")
                continue

        adjusted = trimmed
        adjustedCases[state_name] = adjusted
        adjusted_dom = trimmed_dom
        adjustedCases_dom[state_name] = adjusted_dom
    
        # Include uncertainty of adjustment on onset on right side because we don't know future confirmed cases to compute   recent onset cases
        # t: target date
        # T: latest date
        # L: delay limit, min(L; cumulative(p_infection_onset_delay(0 to L)) > threshould)
        P_DELAY_THRESHOLD = 0.99
        lastDate = adjusted_dom.index[-1][1]
        delayOffsetDays = (obsDate - lastDate).days
        #p_infection_onset_delay_offsetted = p_infection_onset_delay[delayOffsetDays:]
        L = np.argmax(np.cumsum(p_infection_onset_delay) > P_DELAY_THRESHOLD) 
        no_need_adjust = adjusted_dom[:-L].values
        stdCases = []
        if len(no_need_adjust) >= L:
            for lag in range(1+delayOffsetDays, L):
                stdCases.append(np.std(no_need_adjust[lag:] - no_need_adjust[:-lag]))
    
        result = {}

        result['adjustedCases'] = adjusted
        result['adjustedDomesticCases'] = adjusted_dom
        result['confirmedOnlyCases'] = confirmedOnly
        result['confirmedOnlyDomesticCases'] = confirmedOnly_dom
    
        # store std from uncertainty of future confirmed cases for later adjustment
        result['std_future_cases'] = stdCases
        #result['std_future_rt'] = stds_case_ratios
    
        # Holds all posteriors with every given value of sigma
        result['posteriors'] = []
        
        # Holds the log likelihood across all k for each value of sigma
        result['log_likelihoods'] = []
        
        for sigma in sigmas:
            posteriors, log_likelihood = get_posteriors(adjusted, adjusted_dom, singleTau=singleTau, sumStyle=sumStyle, includePosterior=includePosterior, sigma=sigma)
            result['posteriors'].append(posteriors)
            result['log_likelihoods'].append(log_likelihood)
        
        # Store all results keyed off of state name
        results[state_name] = result
    
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
    
    # modify posteriors to include the uncertainty due to future confirmed cases
    for state_name, result in results.items():
        posteriors = result['posteriors'][max_likelihood_index]
        most_likely = posteriors_get_max_point(posteriors)
        #stds_future_rt = result['std_future_rt']
    
        # simulate Rt with bump on future confirmed case
        most_likely_bumpeds = []
        adjusted0 = adjustedCases[state_name]
        adjusted_dom0 = adjustedCases_dom[state_name]
    
        lastDate = max(adjusted0.index[-1][1], adjusted_dom0.index[-1][1])
        delayOffsetDays = (obsDate - lastDate).days
        p_infection_onset_delay_offsetted = p_infection_onset_delay[delayOffsetDays:]
        L = np.argmax(np.cumsum(p_infection_onset_delay) > P_DELAY_THRESHOLD) - delayOffsetDays
    
        # if series is not long enough to compute stdCases (thus Null),
        # the series is too short and not statistically meaningful to calculate stdCases
        # So we do not make adjustment for uncertaity of future cases modulation
        # This no-treating strategy should be alright because confidence interval from Poisson/gamma distribution is    already large
        stdCases = result['std_future_cases']
        if stdCases:
            bump_size = max(stdCases)
            for d in range(1, L):
                if len(adjusted0) - len(p_infection_onset_delay_offsetted) + d > 0:
                    p_delay_values = np.concatenate((p_infection_onset_delay_offsetted[d:], np.zeros(len(adjusted0) - len   (p_infection_onset_delay_offsetted) + d)))
                else:
                    p_delay_values = np.resize(p_infection_onset_delay_offsetted[d:], (1, len(adjusted0)))[0]
                bumpedAdjusted = adjusted0 + bump_size * p_delay_values[::-1] # +1 of the confirmed cases of d-day later
                bumpedAdjusted_dom = adjusted_dom0 + bump_size * p_delay_values[::-1] # +1 of the confirmed cases of d-day   later
                bumpedPosteriors, dummy = get_posteriors(bumpedAdjusted, bumpedAdjusted_dom, singleTau=singleTau, sumStyle=sumStyle, includePosterior=includePosterior, sigma=sigma)
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
        
                #posteriorsfilename_before = posterior_to_be_modified.name[0] + posterior_to_be_modified.name       [1].strftime('%Y-%m-%d') + ".csv"
                #posterior_to_be_modified.to_csv(posteriorsfilename_before)
                #posteriorsfilename_after = posterior_to_be_modified.name[0] + posterior_to_be_modified.name        [1].strftime('%Y-%m-%d') + "_after.csv"
                #np.savetxt(posteriorsfilename_after, posterior_modified, delimiter=",")
                #normalfilename = posterior_to_be_modified.name[0] + posterior_to_be_modified.name[1].strftime  ('%Y-%m-%   d') + "_normal_dist.csv"
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
        adjustedCases = result["adjustedCases"].rename("adjustedCases")
        adjustedDomesticCases = result["adjustedDomesticCases"].rename("adjustedDomesticCases")
        confirmedOnlyCases = result["confirmedOnlyCases"].rename("confirmedOnlyCases")
        confirmedOnlyDomesticCases = result["confirmedOnlyDomesticCases"].rename("confirmedOnlyDomesticCases")
        result = pd.concat([most_likely, hdis_90, hdis_50, adjustedCases, adjustedDomesticCases, confirmedOnlyCases, confirmedOnlyDomesticCases], axis=1)
        if final_results is None:
            final_results = result
        else:
            final_results = pd.concat([final_results, result])
    
    print('Done final result.')
    
    #Save final result to csv files
    for state_name, confirmedOnly in statesConfirmedlOnly_to_process.groupby("state"):
        filename = "data/" + state_name + ".csv"
        if os.path.isfile(filename):
            os.remove(filename)
        if final_results is not None and state_name in final_results.index:
            final_result = final_results.filter(like=state_name, axis=0)
            final_result.loc[state_name].to_csv(filename)
    #for state_name, final_result in final_results.groupby("state"):
    #    final_result.loc[state_name].to_csv("data/" + state_name + ".csv")
    print('Saved final result.')

