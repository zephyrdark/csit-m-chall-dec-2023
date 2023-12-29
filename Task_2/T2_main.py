# Learning Resources
## https://www.kaggle.com/code/jaseemck/audio-processing-using-librosa-for-beginners
## https://www.pnsn.org/spectrograms/what-is-a-spectrogram

import librosa
import matplotlib.pyplot as plt
import numpy as np

a = "./Task_2/T2_audio_a.wav"
b = "./Task_2/T2_audio_b.wav"
c = "./Task_2/T2_audio_c.wav"
d = "./Task_2/T2_audio_d.wav"

ya, sra = librosa.load(a,sr=librosa.get_samplerate(a))
yb, srb = librosa.load(b,sr=librosa.get_samplerate(b))
yc, src = librosa.load(c,sr=librosa.get_samplerate(c))
yd, srd = librosa.load(d,sr=librosa.get_samplerate(d))

fig, ax = plt.subplots(nrows=4,ncols=1,sharex=True)
Da = librosa.amplitude_to_db(np.abs(librosa.stft(ya)),ref=np.max)
librosa.display.specshow(Da,y_axis="linear",x_axis="time",sr=sra,ax=ax[0])
Db = librosa.amplitude_to_db(np.abs(librosa.stft(yb)),ref=np.max)
librosa.display.specshow(Db,y_axis="linear",x_axis="time",sr=srb,ax=ax[1])
Dc = librosa.amplitude_to_db(np.abs(librosa.stft(yc)),ref=np.max)
librosa.display.specshow(Dc,y_axis="linear",x_axis="time",sr=src,ax=ax[2])
Dd = librosa.amplitude_to_db(np.abs(librosa.stft(yd)),ref=np.max)
librosa.display.specshow(Dd,y_axis="log",x_axis="time",sr=srd,ax=ax[3])

fig.savefig("./Task_2/output.png")

# T2_audio_a.wav = MINS
# T2_audio_b.wav = NOON
# T2_audio_c.wav = SEVEN
# T2_audio_d.wav = PAST
# answer in HH:MM = Seven mins past noon ; 12:07