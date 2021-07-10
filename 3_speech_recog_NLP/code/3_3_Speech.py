import numpy
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import soundfile #Install as pysoundfile, Run the Command: pip install PySoundFile


from scipy import signal
from scipy.fft import fftshift

wavPath='..\\alphabetwav\\a_a.wav'
soundArr, soundSampleRate = soundfile.read(wavPath)
print("The shape of the audio wav file is " + str(soundArr.shape))
print("The sampling frequency is Fs=" + str(soundSampleRate) + " Hz")
numS = soundArr.shape[0] // soundSampleRate
print("The sample is {} seconds long".format(numS))
print("Or {:.2f} minutes".format(numS / 60))
fig, ax = plt.subplots(figsize = (20, 7))
ax.plot(soundArr[:273056])
plt.ylabel('Signal Strength')
plt.xlabel('Samples')
plt.show()

fs=soundSampleRate
soundL=soundArr[:,0]
soundR=soundArr[:,1]



def cal_cosine_similarity():
  a_embedding_np=numpy.load("Archive\\audio_embedding_a_a.npy")
  m_embedding_np=numpy.load("Archive\\audio_embedding_m_m.npy")
  cos_simi=pd.DataFrame(cosine_similarity(a_embedding_np,m_embedding_np))
  eu=pd.DataFrame(euclidean_distances(a_embedding_np,m_embedding_np))
  return cos_simi


def cal_heatmap(cos_simi):
  uniform_data = cos_simi
  ax = sns.heatmap(uniform_data, linewidth=0.5)
  plt.show()
  plt.figure(figsize=(200, 100))

def cal_spectogram(soundL,fs):
  f, t, Sxx = signal.spectrogram(soundL, fs)
  plt.figure(figsize=(20, 10))
  plt.pcolormesh(t, f, Sxx, shading='gouraud')#, cmap=plt.cm.Pastel1)
  #f, t, Sxx = signal.spectrogram(soundR, fs)
  #plt.pcolormesh(t, f, Sxx,cmap=plt.cm.gray)
  plt.ylim(1,1500)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.show()
  #plt.figure(figsize=(20, 40))


cos_simi=cal_cosine_similarity()
print(cos_simi)
cal_heatmap(cos_simi)
cal_spectogram(soundL,fs)
