import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft
from pydub import AudioSegment, effects

# librosa - load sample into array
sample = "./Task_3/C.Noisy_Voice.wav"
y, sr = librosa.load(sample,sr=librosa.get_samplerate(sample),duration=8)

# matplotlib, librosa - plot as waveform
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.waveshow(y, sr=sr, ax=ax[0], color="blue")
ax[0].set(title='Envelope view, mono')
ax[0].label_outer()

# librosa - plot as spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)),ref=np.max)
librosa.display.specshow(D,y_axis="linear",x_axis="time",sr=sr,ax=ax[1])
fig.savefig("./Task_3/output0.png")

# matplotlib, scipy - do fast fourier transform & plot
N = sr * 8
yf = rfft(y)
xf = rfftfreq(N, 1 / sr)
fig, ax = plt.subplots(nrows=1, sharex=True)
plt.plot(xf, np.abs(yf))
fig.savefig("./Task_3/output1.png")

# matplotlib - filter to remove frequency of above 2000 Hz
## maximum frequency is half the sample rate
points_per_freq = len(xf) / (sr / 2)
## target frequency is all above 2000 Hz
target_idx = int(points_per_freq * 2000)
yf[target_idx + 1 : target_idx + int(points_per_freq * 10000)] = 0
fig, ax = plt.subplots(nrows=1, sharex=True)
plt.plot(xf, np.abs(yf))
fig.savefig("./Task_3/output2.png")

# matplotlib - inverse rfft
new_y = irfft(yf)
fig, ax = plt.subplots(nrows=1, sharex=True)
plt.plot(new_y)
fig.savefig("./Task_3/output3.png")
sf.write("./Task_3/output.wav", new_y, sr, subtype='PCM_24')

#pydub - normalise volume
rawsound = AudioSegment.from_file("./Task_3/output.wav", "wav")  
normalizedsound = effects.normalize(rawsound)  
normalizedsound.export("./Task_3/output2.wav", format="wav")

