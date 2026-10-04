import numpy as np 
import IPython.display as ipd
import matplotlib.pyplot as plt

    
# complete the function k_harmonics  
def harmonics_arpeggiator(r, k, freq, dur, sample_rate): 
    t = np.arange(0, dur, 1.0 / sample_rate)
    note_size = t.shape[0]

    # array to hold single arpeggio cycle
    cycle = np.zeros(k * note_size)

    # build arpeggio
    for i in range(k):

      h_freq = freq * (1 + i)
      
      note = np.sin(2 * np.pi * h_freq * t)

      # determine start and stop index
      start_index = i * note_size
      end_index = start_index + note_size

      # insert note into desired time slot in arpeggio cycle
      cycle[start_index:end_index] = note

    # loop this cycle r times
    return np.tile(cycle, r)




# use the code in a notebook cell to plot/listen to the resulting 
# signal of the k_harmonics function 
f0 = 220
sr = 44100 
duration = 0.125
r = 4 
k = 12 

signal = harmonics_arpeggiator(r, k, f0, duration, sr)

# use in notebook for listening to audio 
ipd.Audio(signal, rate=sr)

