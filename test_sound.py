
import wave
import matplotlib.pyplot as plt
import numpy as np
import math as ma
import librosa
import librosa.display
import IPython.display as ipd

fname = 'sound_src/Multimedia3.wav'
version = 'v2'
fsize = 256
wave, fs = librosa.load(fname)

# hop:移動單位, frame:處理單位
hop = int(0.005 * fs) # 5 毫秒
frame = int(0.02 * fs)# 20毫秒

ipd.Audio(fname) # load a local WAV file

plt.figure(figsize=(25, 5))
librosa.display.waveplot(wave, sr=fs)
plt.title('Waveform Contour')
plt.savefig("output/waveform_" + str(version) + ".png")
# plt.show()
plt.clf()
plt.close()


# zcr
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
        zcr[i] = sum(cache) / len(cache)
    return zcr

zcr = zero_crossing_rate(wave, frame, frame - hop)
time = np.arange(0, len(zcr)) * (len(wave) / len(zcr) / fs)
plt.figure(figsize = (25,5))
plt.plot(time, zcr)
plt.xlabel('time (sec)')
plt.ylabel('Zero Crossing Rate Contour')
plt.title('ZCR')
plt.savefig("output/ZCR_" + str(version) + ".png")
#plt.show()
plt.clf()
plt.close()

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
        testdata = wave[interval[0]:interval[-1]+1]

        wkernel = np.power(np.hamming(fsize), 2)
        wkernel = wkernel / sum(wkernel)
        hamming_input = np.power(testdata, 2)

        ste[i] = sum(np.convolve(wkernel, hamming_input, mode = 'same'))

        #ste[i] = sum( np.multiply( np.power(testdata, 2), np.power(hamming_window(interval, interval[0], len(testdata)), 2) )) 
        #ste[i] = sum( np.power(testdata, 2))
    return ste

ste = short_time_energy(wave, frame, frame - hop)
time = np.arange(0, len(ste)) * (len(wave) / len(ste) / fs)
plt.figure(figsize = (25 ,5))
plt.plot(time, ste)
plt.xlabel('time (sec)')
plt.ylabel('energy')
plt.title('Energy Contour')
plt.savefig("output/STE_" + str(version) + ".png")
# plt.show()
plt.clf()
plt.close()


# 用librosa套件驗證 zcr, ste 
lib_zcr = librosa.feature.zero_crossing_rate(wave, frame_length = frame, hop_length = hop)[0]
lib_ste = librosa.feature.rms(wave, frame_length = frame, hop_length = hop)[0]

plt.figure(figsize = (12 ,6))

plt.subplot(3, 2, 1)
zcr_ = librosa.util.normalize(zcr)
time = np.arange(0, len(zcr_)) * (len(wave) / len(zcr_) / fs)
plt.plot(time, zcr_, label = 'my zero crossing rate')
plt.title('My Zero Crossing Rate Contour')
plt.xlabel('time (sec)')
plt.ylabel('my zcr (normalized)')

plt.subplot(3, 2, 2)
lib_zcr = librosa.util.normalize(lib_zcr)
time = np.arange(0, len(lib_zcr)) * (len(wave) / len(lib_zcr) / fs)
plt.plot(time, lib_zcr, label = 'lib zero crossing rate Contour')
plt.title('Librosa Zero Crossing Rate')
plt.xlabel('time (sec)')
plt.ylabel('librosa zcr (normalized)')

plt.subplot(3, 2, 5)
ste_ = librosa.util.normalize(ste)
time = np.arange(0, len(ste_)) * (len(wave) / len(ste_) / fs)
plt.plot(time, ste_, label = 'my short time energy')
plt.title('My Energy Contour')
plt.xlabel('time (sec)')
plt.ylabel('my ste (normalized)')

plt.subplot(3, 2, 6)
lib_ste = librosa.util.normalize(lib_ste)
time = np.arange(0, len(lib_ste)) * (len(wave) / len(lib_ste) / fs)
plt.plot(time, lib_ste, label = 'lib short time energy')
plt.title('Librosa Energy Contour')
plt.xlabel('time (sec)')
plt.ylabel('librosa ste (normalized)')

# plt.show()
plt.clf()
plt.close()

# pitch contour

# 用現成函數抓出音高
frames = librosa.util.frame(wave, frame_length = frame, hop_length = frame)
pitches, magnitudes = librosa.core.piptrack(wave, sr = fs, hop_length = hop, threshold = 0.1)



# 留瞬間最高頻
def max_pitch(pitches, shape):
    max_pitches = []
    for i in range(0, shape[1]):
        max_pitches.append(np.max(pitches[:,i]))
    return max_pitches

# window function處理，使用卷積函數
def window_function_conversion(pitch_input ,wlen = 12 ,window='hanning'):
        if wlen < 3:
                return pitch_input
        assert window in ['rect','hanning','hamming','blackman','bartlett'], 'window input error'
        if window == 'rect': 
            #rectangular
            wkernel = np.ones(wlen,'d')
        else:
            # function impliment
            # hanning, hamming, blackman, bartlett
            wkernel = eval('np.'+window+'(wlen)')

        output = np.convolve( wkernel / wkernel.sum(), pitch_input, mode ='same')
        return output[wlen : -wlen + 1]

original_pitch_data = max_pitch(pitches, pitches.shape)
plt.figure(figsize=(20, 4))
plt.plot(original_pitch_data)
plt.title("Pitch Contour (original)")
plt.xlabel('time (sec)')
plt.ylabel('pitch (hz)')
plt.savefig('output/pitch/PC_original_' + str(version) +'.png')
# plt.show()
plt.clf()
plt.close()

# 探討各個不同的window對波形的影響
window_str = ['hanning','hamming','blackman','bartlett']
plt.figure(figsize = (12, 6))
for i in range(4):
    smooth_pitch_data = window_function_conversion(original_pitch_data, window = window_str[i])
    plt.subplot(3, 2, i % 2 + 4 * (i // 2) + 1)
    plt.plot(smooth_pitch_data)
    plt.title('Pitch Contour (' + window_str[i] + ')')
    plt.xlabel('time (sec)')
    plt.ylabel('pitch (hz)')

plt.savefig('output/pitch/PC_smooth_' + str(version) +'.png')
# plt.show()
plt.clf()
plt.close()

# end point detection

def ITL_below_ITU(value, ITL, start_end):
    ITL = list(ITL)
    test = ITL.index(value)
    diff = 1
    assert start_end == 1 or start_end == 0, "start_end 只能是1或0"
    if start_end == 0:
        while diff == 1 and test - 1 >= 0:
            if ITL[test] - ITL[test - 1] != 1:
                return ITL[test]
            test -= 1
    else :
        while diff == 1 and test + 1 < len(ITL):
            if ITL[test + 1] - ITL[test] != 1:
                return ITL[test]
            test += 1
    return test

def end_point_detect(ste, fsize, upper, lower):
    ITU = np.where( ste > upper )[0]
    ITL = np.where( ste > lower )[0]

    iterations = len(ITU)
    ed_start = []
    ed_end = []

    ed_start.append(ITL_below_ITU(ITU[0], ITL, 0))
    for i in range(iterations - 1) :
        if ITU[i + 1] - ITU[i] != 1:
            ed_end.append(ITL_below_ITU(ITU[i], ITL, 1))
            ed_start.append(ITL_below_ITU(ITU[i + 1], ITL, 0))
    ed_end.append(ITL_below_ITU(ITU[-1], ITL, 1))

    assert len(ed_end) == len(ed_start), "inconsistent length"
    ed_start = np.array(ed_start)
    ed_end = np.array(ed_end)
    convert_to_sec = ma.ceil(len(wave) / len(ste)) / fs
    ed_start = ed_start * convert_to_sec
    ed_end = ed_end * convert_to_sec
    return ed_start, ed_end

start_time, end_time = end_point_detect(ste, fsize, upper = 1.2, lower = 0.1)

t = np.linspace(0,len(wave) / fs, len(wave))
plt.figure(figsize = (25,5))
plt.plot(t, wave, label = "waveform")
first = True
for start, end in zip(start_time, end_time):
    plt.axvline(start, color = '#01B468', label = 'start' if first else None)
    plt.axvline(end  , color = '#FF359A', label = 'end' if first else None)
    first = False
plt.xlabel('time (sec)')
plt.ylabel('waveform')
plt.title('End Point Detection')
plt.legend(loc='upper right')
plt.savefig('output/EPD_' + str(version) +'.png')
# plt.show()
plt.clf()
plt.close()






    








        




