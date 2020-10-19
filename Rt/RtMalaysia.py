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

from datetime import date
from datetime import datetime

from computeRt import computeRt

# Load case file
url = 'data/casesForRt.csv'
states = pd.read_csv(url,
                     usecols=['date', 'state', 'cases'],
                     parse_dates=['date'],
                     index_col=['state', 'date'],
                     squeeze=True).sort_index()

# Configuration
isCaseCumulative = True
revert_to_confirmed_base = False
rightCensorshipByDelayFunctionDevision = False
singleTau = False
sumStyle = "Nishiura" # "Exponential", "K-Sys"
includePosterior = True
backProjection = True
#leftUncertaintyStyle = "Nothing"
leftUncertaintyStyle = "ConfirmedTotal"
#leftUncertaintyStyle = "ConfirmedDecomposed"
#FILTERED_REGION_CODES = ['Malacca']

#FILTERED_REGION_CODES = ['Malaysia']
#filterInclusive = True
FILTERED_REGION_CODES = ['']
filterInclusive = False

p_EPS = 0.001
#p_onset_comfirmed_delay = pd.read_csv("data/onset_confirmed_delay.csv", index_col=None, header=None, squeeze=True)
P_CONFIRMED_DELAY_T = np.linspace(0, 30, 31)
p_onset_comfirmed_delay_cum = sps.weibull_min(c = 1.741, scale = 8.573).cdf(x = P_CONFIRMED_DELAY_T)
p_onset_comfirmed_delay = p_onset_comfirmed_delay_cum[1:] - p_onset_comfirmed_delay_cum[:-1]
p_onset_comfirmed_delay = np.insert(p_onset_comfirmed_delay, 0, p_EPS)

P_INFECTION_ONSET_DELAY_T = np.linspace(0, 30, 31)
p_infection_onset_delay_cum = sps.lognorm(scale = math.exp(1.519), s = 0.615).cdf(x = P_INFECTION_ONSET_DELAY_T)
p_infection_onset_delay = p_infection_onset_delay_cum[1:] - p_infection_onset_delay_cum[:-1]
p_infection_onset_delay = np.insert(p_infection_onset_delay, 0, p_EPS)
p_infection_confirm_delay = np.convolve(p_onset_comfirmed_delay, p_infection_onset_delay)

computeRt(statesOnset = [], statesOnset_dom = [], statesConfirmedOnly = states, statesConfirmedOnly_dom = states, isCaseCumulative = isCaseCumulative, p_onset_comfirmed_delay = p_onset_comfirmed_delay, p_infection_onset_delay = p_infection_onset_delay, p_infection_confirm_delay = p_infection_confirm_delay, includePosterior = includePosterior, sumStyle = sumStyle, rightCensorshipByDelayFunctionDevision = rightCensorshipByDelayFunctionDevision, singleTau = singleTau, FILTERED_REGION_CODES = FILTERED_REGION_CODES, filterInclusive = filterInclusive, revert_to_confirmed_base = revert_to_confirmed_base, backProjection = backProjection, leftUncertaintyStyle = leftUncertaintyStyle)
