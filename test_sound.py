
import wave
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import librosa
import librosa.display
import IPython.display as ipd

fname = 'Multimedia2.wav'

wave, fs = librosa.load(fname)

ipd.Audio(fname) # load a local WAV file

plt.figure(figsize=(25, 5))
librosa.display.waveplot(wave, sr=fs)
plt.title('waveform')
plt.show()

#zcr

def sgn(l):
    ans = []
    for i in range(len(l)):
        ans.append(1 if l[i] >= 0 else 0)
    return ans

def zero_crossing_rate(wave, fsize, overlapfsize):
    wsize = fsize - overlapfsize
    testnum = ma.ceil(len(wave) / wsize)
    zrc = np.zeros((testnum,1))
    for i in range(testnum):
        testdata = wave[i * wsize : min(i * wsize + fsize, len(wave))]
        cache = abs(np.array(sgn(testdata[1::])) - np.array(sgn(testdata[0:-1]))) 
        zrc[i] = sum(cache) / (2 * len(testdata))
    return zrc

zrc = zero_crossing_rate(wave, 256, 0)
time = np.arange(0, len(zrc)) * (len(wave) / len(zrc) / fs)
plt.figure(figsize = (25,5))
plt.plot(time, zrc)
plt.xlabel('time (sec)')
plt.ylabel('zero crossing rate')
plt.title('ZRC')
plt.show()

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
        #ste[i] = sum( np.multiply( np.power(testdata, 2), np.power(hamming_window(interval, interval[0], len(testdata)), 2) )) 
        ste[i] = sum( np.power(testdata, 2))
    return ste

ste = short_time_energy(wave, 256, 0)
time = np.arange(0, len(zrc)) * (len(wave) / len(zrc) / fs)
plt.figure(figsize = (25,5))
plt.plot(time, ste)
plt.xlabel('time (sec)')
plt.ylabel('energy')
plt.title('short time energy')
plt.show()






        




