# Created on Sat Aug 19 14:22:31 2017
# @author: tom

"""
basic FreMEn to find most influential periodicity, call
chosen_period(T, time_frame_sums, time_frame_freqs, longest, shortest, W, ES):
it returns the most influential period in the timeseries, where timeseries are
    the residues between reality and model
where
input: T numpy array Nx1, time positions of measured values
       time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                        over every
                                                        timeframe
       time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                        frequencies(stat)
                                                        in model
                                                        over every
                                                        timeframe
       W numpy array Lx1, sequence of reasonable frequencies
       ES float64, squared sum of squares of residues from the last
                   iteration
and
output: P float64, length of the most influential frequency in default
        units
        W numpy array Lx1, sequence of reasonable frequencies without
                           the chosen one
        ES_new float64, squared sum of squares of residues from
                        this iteration
        dES float64, difference between last and new error

for the creation of a list of reasonable frequencies call
build_frequencies(longest, shortest):
where
longest - float, legth of the longest wanted period in default units,
        - usualy four weeks
shortest - float, legth of the shortest wanted period in default units,
         - usualy one hour.
It is necessary to understand what periodicities you are looking for (or what
    periodicities you think are the most influential)
"""

import numpy as np


def chosen_period(T, time_frame_sums, time_frame_freqs, W, ES,
                  valid_timesteps):
    """
    input: T numpy array Nx1, time positions of measured values
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                            frequencies(stat)
                                                            over every
                                                            timeframe
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from the last
                       iteration
    output: P float64, length of the most influential frequency in default
            units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
            ES_new float64, squared sum of squares of residues from
                            this iteration
            dES float64, difference between last and new error
    uses: np.sum(), np.max(), np.absolute()
          complex_numbers_batch(), max_influence()
    objective: to choose the most influencing period in the timeseries, where
               timeseries are the residues between reality and model
    """
    S = (time_frame_sums - time_frame_freqs)[valid_timesteps]
    ES_new = (np.sum(S ** 2)) ** 0.5
    #print('squared sum of squares of residues: ' + str(ES_new))
    if ES == -1:
        #print('difference in errors: ' + str(-ES_new))
        dES = -ES_new
    else:
        dES = ES_new - ES
        #print('difference in errors: ' + str(dES))
    G = complex_numbers_batch(T, S, W)
    P, W = max_influence(W, G)
    sum_of_amplitudes = np.sum(np.absolute(G))
    return P, W, ES_new, sum_of_amplitudes  # dES


def build_frequencies(longest, shortest):
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
    output: W numpy array Lx1, sequence of frequencies
    uses: np.arange()
    objective: to find frequencies w_0 to w_k
    """
    k = int(longest / shortest) + 1
    W = np.float64(np.arange(k)) / float(longest)
    return W


def complex_numbers_batch(T, S, W):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, sequence of measured values
           W numpy array Lx1, sequence of reasonable frequencies
    output: G numpy array Lx1, sequence of complex numbers corresponding
            to the frequencies from W
    uses: np.e, np.newaxis, np.pi, np.mean()
    objective: to find sparse(?) frequency spectrum of the sequence S
    """
    Gs = S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2))
    G = np.mean(Gs, axis=1)
    return G


def max_influence(W, G):
    """
    input: W numpy array Lx1, sequence of reasonable frequencies
           G numpy array Lx1, sequence of complex numbers corresponding
                              to the frequencies from W
    output: P float64, length of the most influential frequency in default
                       units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
    uses: np.absolute(), np.argmax(), np.float64(),np.array()
    objective: to find length of the most influential periodicity in default
               units and return changed list of frequencies
    """
    #### testovani zmeny "sily" periody pri zmene poctu shluku
    #print('soucet velikosti period')
    #print(np.sum(np.absolute(G)))
    #### konec testovani
    #maximum_position = np.argmax(np.absolute(G[1:])) + 1
    maximum_position = np.argmax(np.absolute(G))
    #print('velikost nejvlivnejsi periody')
    #print(np.max(np.absolute(G)))
    # ! probably not elegant way of changing W
    #WW = list(W)
    #influential_frequency = WW.pop(maximum_position)
    #W = np.array(WW)
    # !
    influential_frequency = W[maximum_position]
    if influential_frequency == 0 or np.isnan(np.max(np.absolute(G))):
        # nekdy to pada, nevim, proc, tak jsem dal druhou podminku: prezkoumat!
        P = np.float64(0.0)
    else:
        P = 1 / influential_frequency
    #print('vybrana perioda')
    #print(P)
    return P, W
