import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft

# load sample into array
sample = "/home/alvin/repos/csit-m-chall-dec-2023/Task_3/C.Noisy_Voice.wav"
y, sr = librosa.load(sample,sr=librosa.get_samplerate(sample),duration=8)

# plot as waveform
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.waveshow(y, sr=sr, ax=ax[0], color="blue")
ax[0].set(title='Envelope view, mono')
ax[0].label_outer()

# plot as spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)),ref=np.max)
librosa.display.specshow(D,y_axis="linear",x_axis="time",sr=sr,ax=ax[1])
fig.savefig("/home/alvin/repos/csit-m-chall-dec-2023/Task_3/output0.png")

# do fast fourier transform & plot
N = sr * 8
yf = rfft(y)
xf = rfftfreq(N, 1 / sr)
fig, ax = plt.subplots(nrows=1, sharex=True)
plt.plot(xf, np.abs(yf))
fig.savefig("/home/alvin/repos/csit-m-chall-dec-2023/Task_3/output1.png")

# The maximum frequency is half the sample rate
points_per_freq = len(xf) / (sr / 2)
# Our target frequency is all above 2000 Hz
target_idx = int(points_per_freq * 2000)
yf[target_idx + 1 : target_idx + int(points_per_freq * 10000)] = 0
fig, ax = plt.subplots(nrows=1, sharex=True)
plt.plot(xf, np.abs(yf))
fig.savefig("/home/alvin/repos/csit-m-chall-dec-2023/Task_3/output2.png")

#inverse rfft
new_y = irfft(yf)
fig, ax = plt.subplots(nrows=1, sharex=True)
plt.plot(new_y)
fig.savefig("/home/alvin/repos/csit-m-chall-dec-2023/Task_3/output3.png")
sf.write("/home/alvin/repos/csit-m-chall-dec-2023/Task_3/output.wav", new_y, sr, subtype='PCM_24')

# print("yf",yf.shape, max(yf), min(yf))
# print("xf",xf.shape, max(xf), min(xf))

# def trimmer(iarr):
#     oarr=[]
#     for i in iarr:
#         if i >= 2000 or i <= -2000:
#             print(i)
#             oarr.append(0)
#         else:
#             oarr.append(i)
#     print("oarr",len(oarr))
#     return oarr

# txf = trimmer(xf)
# plt.plot(yf, np.abs(txf))
# fig.savefig("/home/alvin/repos/csit-m-chall-dec-2023/Task_3/output4.png")

# print("txf",len(txf), max(txf), min(txf))
# # print("tyf",len(tyf), max(tyf), min(tyf))
