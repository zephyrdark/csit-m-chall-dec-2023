import librosa
import soundfile

samplePath = "./Task_1/T1_audio.wav"
sampleRate = librosa.get_samplerate(samplePath)
print(sampleRate)

sampleArray = librosa.load(samplePath,sr=None)
print("sampleArray = ",sampleArray[0],sampleArray[1])
sr = sampleArray[1]
reverseSampleArray = sampleArray[0][::-1]
spedSampleArray = librosa.effects.time_stretch(reverseSampleArray, rate=2.0)

soundfile.write("./Task_1/output.wav", spedSampleArray, sr, subtype='PCM_24')

## Latitude: North 67.9222, Longitude: East 26.5046
## 67°55'19.9"N 26°30'16.6"E
## answer(?) = 99600 Sodankylä, Finland