import numpy as np
from ising import *

class Ising_analysis:
    def __init__(self,sim_list):
        self.sim_list = sim_list
        self.m_rav = None
        self.av_methyl = None

    def average_methylation(self):
        """
        Calculates average methylation rate for each time step across the whole region.
        :return:
        """
        self.av_methyl = np.array([sim.m_save.mean(axis=1) for sim in self.sim_list])

    def rolling_average_methylation(self,window=5):
        """
        Smooths the methylation profile.
        Kind of pointless..
        :param window:
        :return:
        """
        self.m_rav = rolling_average_timeseries(self.sim.m_save,window=window)



@jit(nopython=True)
def rolling_average(m,window):
    window_span = 2*window-1
    m_av = np.zeros_like(m,dtype=np.float64)
    for i in range(window_span):
        m_av += np.roll(m,i - window+1)
    m_av = m_av/window_span
    return m_av

@jit(nopython=True)
def rolling_average_timeseries(m_save,window):
    m_rav = np.zeros(m_save.shape, dtype=np.float64)
    for i, m in enumerate(m_save):
        m_rav[i] = rolling_average(m,window)
    return m_rav