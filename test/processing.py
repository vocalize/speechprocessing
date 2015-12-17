import speechprocessing.processing
import wave

def create_test_file():
  noise_output = wave.open('noise.wav', 'w')
  noise_output.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))

def setup_module(module):
  create_test_file()