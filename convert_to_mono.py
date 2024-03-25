from os import listdir
import librosa
import numpy as np
import soundfile
from scipy.io import wavfile

files = listdir(".\\tts\\soundbites\\")
for file in files:
	pathf = "./tts/soundbites/"+file
	print(pathf)
	y, sr = librosa.load(pathf, sr=44100)
	soundfile.write(pathf, librosa.to_mono(y), 44100)