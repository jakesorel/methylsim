from ising import Ising
import numpy as np
import matplotlib.pyplot as plt
from ising_analysis import Ising_analysis

sim = Ising(300)
sim.generate_init(0.2)
sim.make_conv_kernels(2)
print(sim.kern)

## if the convolution window is 2 (i.e. only direct neighbours, then there are 8 possibilities for a transition)
##Assign the rates below
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

##These conditional probabilities sum to give the total probability of a methylation or demethylation event.
##For example, if neighbourhood is irrelevant, then the first four probabilities should be the same. ditto the last four
##For parameter scans, it's probably worth pinning the mirror image 'states' to the same rates, unless there's a biological reason for directionality at this microscopic level (i can't think of one).


sim.assign_transition_rates(rates)
sim.set_t_span(dt=0.1,tfin=10000)
sim.simulate(skip=100) ##skip defines the number of iterations of the algorithm to ignore while saving (r.e. memory)


##Plot the results. Black = methylated.
fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(sim.m_save,aspect="auto",interpolation="None",extent=[0,sim.N,sim.tfin,0],cmap=plt.cm.Greys)
ax.set(xlabel="Methylation site",ylabel="Time")
fig.subplots_adjust(top=0.8,bottom=0.3,left=0.3,right=0.8)
fig.show()


