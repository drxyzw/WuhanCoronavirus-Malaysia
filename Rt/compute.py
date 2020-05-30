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

#from matplotlib import pyplot as plt
#from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
#from matplotlib.dates import date2num
#import matplotlib.dates as mdates
#import matplotlib.ticker as ticker

from datetime import date
from datetime import datetime

FILTERED_REGION_CODES = ['']
#FILTERED_REGION_CODES = ['Kedah']

# Load case file
url = 'data/casesForRt.csv'
states = pd.read_csv(url,
                     usecols=['date', 'state', 'cases'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()

revert_to_confirmed_base = False

# Ensure all case diffs are greater than zero
for state, grp in states.groupby('state'):
    new_cases = grp.diff().dropna()
    is_positive = new_cases.ge(0)
    
    try:
        assert is_positive.all()
    except AssertionError:
        print(f"Warning: {state} has date with negative case counts")
        print(new_cases[~is_positive])

#Download from web - already done, so no need
#def download_file(url, local_filename):
#    """From https://stackoverflow.com/questions/16694907/"""
#    with requests.get(url, stream=True) as r:
#        r.raise_for_status()
#        with open(local_filename, 'wb') as f:
#            for chunk in r.iter_content(chunk_size=8192): 
#                if chunk: # filter out keep-alive new chunks
#                   f.write(chunk)
#    return local_filename
#
##URL = "./data/latestdata.csv"
#URL = "https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.csv"
#LINELIST_PATH = 'data/linelist.csv'
#
#if not os.path.exists(LINELIST_PATH):
#    print('Downloading file, this will take a while ~100mb')
#    try:
#        download_file(URL, LINELIST_PATH)
#        clear_output(wait=True)
#        print('Done downloading.')
#    except:
#        print('Something went wrong. Try again.')
#else:
#    print('Already downloaded CSV')
#

#No need to load huge file and create onse-confirmation distribution
###Parse & Clean Patient Info
## Load the patient CSV
#patients = pd.read_csv(
#    'data/linelist.csv',
#    parse_dates=False,
#    usecols=[
#        'date_confirmation',
#        'date_onset_symptoms'],
#    low_memory=False)

#patients.columns = ['Onset', 'Confirmed']

## Only keep if both values are present
#patients = patients.dropna()

## Must have strings that look like individual dates
## "2020.03.09" is 10 chars long
#is_ten_char = lambda x: x.str.len().eq(10)
#patients = patients[is_ten_char(patients.Confirmed) & 
#                    is_ten_char(patients.Onset)]

## Convert both to datetimes
#patients.Confirmed = pd.to_datetime(
#    patients.Confirmed)
#    #patients.Confirmed, format='%Y-%m-%d')
#patients.Onset = pd.to_datetime(
#    patients.Onset)
#    #patients.Onset, format='%Y-%m-%d')

## Only keep records where confirmed > onset
#patients = patients[patients.Confirmed >= patients.Onset]


##Show Relationship between Onset of Symptoms and Confirmation
#ax = patients.plot.scatter(
#    title='Onset vs. Confirmed Dates - COVID19',
#    x='Onset',
#    y='Confirmed',
#    alpha=.1,
#    lw=0,
#    s=10,
#    figsize=(6,6))

#formatter = mdates.DateFormatter('%m/%d')
#locator = mdates.WeekdayLocator(interval=2)

#for axis in [ax.xaxis, ax.yaxis]:
#    axis.set_major_formatter(formatter)
#    axis.set_major_locator(locator)

##plt.show()

##Calculate the Probability Distribution of Delay
## Calculate the delta in days between onset and confirmation
#delay = (patients.Confirmed - patients.Onset).dt.days

## Convert samples to an empirical distribution
#p_delay = delay.value_counts().sort_index()
#new_range = np.arange(0, p_delay.index.max()+1)
#p_delay = p_delay.reindex(new_range, fill_value=0)
#p_delay /= p_delay.sum()

p_delay = pd.read_csv("data/onset_confirmed_delay.csv", index_col=None, header=None, squeeze=True)
## Show our workp
#fig, axes = plt.subplots(ncols=2, figsize=(9,3))
#p_delay.plot(title='P(Delay)', ax=axes[0])
#p_delay.cumsum().plot(title='P(Delay <= x)', ax=axes[1])
#for ax in axes:
#    ax.set_xlabel('days')
#plt.show()

##Select State Data
#state_name = 'Putrajaya'
#
#confirmed = states.xs(state_name).rename(f"{state_name} cases").diff().dropna()
#confirmed.tail()

def confirmed_to_onset(confirmed, p_delay):
    if revert_to_confirmed_base:
        return confirmed
    else:
        assert not confirmed.isna().any()
        
        # Reverse cases so that we convolve into the past
        convolved = np.convolve(confirmed[::-1].values, p_delay)
    
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


#onset = confirmed_to_onset(confirmed, p_delay)

def adjust_onset_for_right_censorship(onset, p_delay):
    if revert_to_confirmed_base:
        return onset, 0
    else:
        cumulative_p_delay = p_delay.cumsum()
        
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


#adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)

#fig, ax = plt.subplots(figsize=(5,3))

#confirmed.plot(
#    ax=ax,
#    label='Confirmed',
#    title=state_name,
#    c='k',
#    alpha=.25,
#    lw=1)

#onset.plot(
#    ax=ax,
#    label='Onset',
#    c='k',
#    lw=1)

#adjusted.plot(
#    ax=ax,
#    label='Adjusted Onset',
#    c='k',
#    linestyle='--',
#    lw=1)

#ax.legend();

##plt.show()


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
GAMMA = 1/7

#Function for Calculating the Posteriors

def get_posteriors(sr, sigma=0.15):
    ## (0) Round adjusted cases for poisson distribution
    #sr = sr.round()

    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1.))

    
    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data = sps.gamma(a=sr[1:]+1., scale=1., loc=0.).pdf(x=lam),
        #data = sps.poisson.pmf(sr[1:].values, lam),
        index = r_t_range,
        columns = sr.index[1:])

    #nStd = 3.;
    #for i in range(1, L):
    #    sr0 = sr[1:][-i]
    #    onset_range = np.linspace(max(0.0, sr0 - nStd*stds[i-1]), sr0 + nStd*stds[i-1], 30)
    #    likelihoods_to_be_integrated = pd.DataFrame(
    #        data = sps.gamma(a=onset_range+1., scale=1., loc=0.).pdf(x=lam[:,-i][:,None]),
    #        index = r_t_range,
    #        columns = onset_range)
    #    normal_dist = sps.norm(loc=sr0,
    #                          scale=stds[i-1]
    #                         ).pdf(onset_range[:, None]) 
    #    likelihood_i = likelihoods_to_be_integrated @ normal_dist
    #    likelihoods[state_name].iloc[:-i] = likelihood_i


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

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]
        
        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        #when sr[i-1] = 0 (then lam(sr[i-1], r_t_range) = 0, so denominator = 0
        #Then we take limit of sr[i] -> +0
        if sr[previous_day] == 0.0:
            numerator = 1. / factorial(sr[current_day]) * np.exp(GAMMA * (r_t_range - 1.) * sr[current_day]) * current_prior
            denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator
        
        # Add to the running sum of log likelihoods
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
sigmas = np.linspace(1/200, 0.1, 20)

targets = ~states.index.get_level_values('state').isin(FILTERED_REGION_CODES)
states_to_process = states.loc[targets]

results = {}
adjustedCases = {}

for state_name, cases in states_to_process.groupby(level='state'):
    #if state_name != 'Johor':
    #if state_name != 'Malaysia' and state_name != 'Johor':
    #    continue

    print(state_name)
    cases = cases.diff().dropna()
    onset = confirmed_to_onset(cases, p_delay)
    adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)
    if revert_to_confirmed_base:
        original, trimmed = prepare_cases_old(adjusted, cutoff=10)
    else:
        trimmed = prepare_cases(adjusted, cutoff=10)
    
    if len(trimmed) == 0:
        if revert_to_confirmed_base:
            original, trimmed = prepare_cases_old(adjusted, cutoff=2)
        else:
            trimmed = prepare_cases(adjusted, cutoff=2)
        if len(trimmed) == 0:
            print(state_name + ": too few cases for statistics")
            continue
    adjusted = trimmed
    adjustedCases[state_name] = adjusted

    # Include uncertainty of adjustment on onset on right side because we don't know future confirmed cases to compute recent onset cases
    # t: target date
    # T: latest date
    # L: delay limit, min(L; cumulative(p_delay(0 to L)) > threshould)
    P_DELAY_THRESHOLD = 0.99
    L = np.argmax(np.cumsum(p_delay) > P_DELAY_THRESHOLD)
    no_need_adjust = adjusted[:-L].values
    stdCases = []
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
        posteriors, log_likelihood = get_posteriors(adjusted, sigma=sigma)
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
    L = np.argmax(np.cumsum(p_delay) > P_DELAY_THRESHOLD)
    most_likely_bumpeds = []
    adjusted0 = adjustedCases[state_name]
    bump_size = 5.0
    for d in range(1, L):
        if len(adjusted0) - len(p_delay) + d > 0:
            p_delay_values = np.concatenate((p_delay.values[d:], np.zeros(len(adjusted0) - len(p_delay) + d)))
        else:
            p_delay_values = np.resize(p_delay.values[d:], (1, len(adjusted0)))[0]
        bumpedAdjusted = adjusted0 + bump_size * p_delay_values[::-1] # +1 of the confirmed cases of d-day later
        bumpedPosteriors, dummy = get_posteriors(bumpedAdjusted, sigma=sigma)
        most_likely_bumped = posteriors_get_max_point(bumpedPosteriors)
        most_likely_change = (most_likely_bumped - most_likely) / bump_size
        most_likely_bumpeds.append(most_likely_change)

    stdCases = result['std_future_cases']
    stdRts = 0. * most_likely
    for d in range(1, L):
        stdRts += most_likely_bumpeds[d-1]**2 * stdCases[d-1]**2
    stdRts = np.sqrt(stdRts)[::-1]
    #stdRts.to_csv("stdRts.csv")

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

        #posteriorsfilename_before = posterior_to_be_modified.name[0] + posterior_to_be_modified.name[1].strftime('%Y-%m-%d') + ".csv"
        #posterior_to_be_modified.to_csv(posteriorsfilename_before)
        #posteriorsfilename_after = posterior_to_be_modified.name[0] + posterior_to_be_modified.name[1].strftime('%Y-%m-%d') + "_after.csv"
        #np.savetxt(posteriorsfilename_after, posterior_modified, delimiter=",")
        #normalfilename = posterior_to_be_modified.name[0] + posterior_to_be_modified.name[1].strftime('%Y-%m-%d') + "_normal_dist.csv"
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



