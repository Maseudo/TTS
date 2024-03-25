from pydub import AudioSegment
from os import listdir

files = listdir(".\\tts\\soundbites\\")
for file in files:
	pathf = "./tts/soundbites/"+file
	print(pathf)
	sound = AudioSegment.from_wav(pathf)
	sound = sound.set_channels(1)
	sound.export(pathf, format="wav")