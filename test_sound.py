
# In[1]
import wave
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import librosa
import librosa.display
import IPython.display as ipd

fname = 'sound_src/Multimedia3.wav'
version = 6

wave, fs = librosa.load(fname)

ipd.Audio(fname) # load a local WAV file

plt.figure(figsize=(25, 5))
librosa.display.waveplot(wave, sr=fs)
plt.title('waveform')
plt.savefig("output/Waveform_" + str(version) + ".png")
#plt.show()

# In[2]

#zcr

def sgn(l):
    ans = []
    for i in range(len(l)):
        ans.append(1 if l[i] >= 0 else 0)
    return ans

def zero_crossing_rate(wave, fsize, overlapfsize):
    wsize = fsize - overlapfsize
    testnum = ma.ceil(len(wave) / wsize)
    zcr = np.zeros((testnum,1))
    for i in range(testnum):
        testdata = wave[i * wsize : min(i * wsize + fsize, len(wave))]
        cache = abs(np.array(sgn(testdata[1::])) - np.array(sgn(testdata[0:-1]))) 
        zcr[i] = sum(cache) / (2 * len(testdata))
    return zcr

zcr = zero_crossing_rate(wave, 256, 0)
time = np.arange(0, len(zcr)) * (len(wave) / len(zcr) / fs)
plt.figure(figsize = (25,5))
plt.plot(time, zcr)
plt.xlabel('time (sec)')
plt.ylabel('zero crossing rate')
plt.title('ZCR')
plt.savefig("output/ZCR_" + str(version) + ".png")
#plt.show()

#short time energy
def hamming_window(m, n, M):
    ans = []
    for i in range(len(m)):
        if m[i] - n >= 0 and m[i] - n <= M - 1:
            ans.append(0.54 - 0.46 * ma.cos(ma.pi * (m[i] - n) / M))
        else :
            ans.append(0)
    return ans

def short_time_energy(wave, fsize, overlapfsize):
    wsize = fsize - overlapfsize
    testnum = ma.ceil(len(wave) / wsize)
    ste = np.zeros((testnum,1))
    interval = []
    for i in range(testnum):
        interval = np.arange(i * wsize , min(i * wsize + fsize, len(wave)))
        testdata = wave[i * wsize : min(i * wsize + fsize, len(wave))]
        ste[i] = sum( np.multiply( np.power(testdata, 2), np.power(hamming_window(interval, interval[0], len(testdata)), 2) )) 
        ste[i] = sum( np.power(testdata, 2))
    return ste

ste = short_time_energy(wave, 256, 0)
time = np.arange(0, len(zcr)) * (len(wave) / len(zcr) / fs)
plt.figure(figsize = (25,5))
plt.plot(time, ste)
plt.xlabel('time (sec)')
plt.ylabel('energy')
plt.title('short time energy')
plt.savefig("output/STE_" + str(version) + ".png")
frame_idxs = np.where( ste > 0.5)[0]
print(frame_idxs)
plt.show()






        




