from ising import Ising
import numpy as np
import matplotlib.pyplot as plt
from ising_analysis import Ising_analysis


##...copied from the single_simulation script.
scale = 10
rates = np.array([0.01, #False False False --> False True False
                  0.05,  #False False  True --> False True  True
                  0.05,  #True False False --> True True False; if actions are symmetric, then should be = to above
                  0.3,  #True False  True --> True True True
                  0.3,  #False  True False --> False False False
                  0.05, #False  True  True --> False  False  True
                  0.05, #True  True False --> True  False False; if actions are symmetric, then should be = to above
                  0.005 #True  True  True --> True  False  True
                  ])*scale

skip = 100
n_rep = 100 ##run the same simulation 100 times.
sim_list = []
for i in range(n_rep):
    sim = Ising(300)
    sim.generate_init(0.2)
    sim.make_conv_kernels(2)
    sim.assign_transition_rates(rates)
    sim.set_t_span(0.1, 10000)
    sim.simulate(skip=skip)
    sim_list.append(sim)

##Note that this could be very easily made parallel.


##This second class (bare bones atm) can be used for analysis of various statistical features of an 'ensemble'
analy = Ising_analysis(sim_list)

##For e.g. can calculate the average methylation across the entire sequence of methylation sites.
analy.average_methylation()
fig, ax = plt.subplots(figsize=(4,3))
for avm in analy.av_methyl:
    ax.plot(analy.sim_list[0].t_span[::skip][:avm.size],avm,color="grey",alpha=0.2,linewidth=0.7)
ax.set(xlim=(0,analy.sim_list[0].t_span.max()),xlabel="Time",ylabel="Average methylation \nfraction")
fig.subplots_adjust(top=0.8,bottom=0.3,left=0.3,right=0.8)

fig.show()
