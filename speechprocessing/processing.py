from __future__ import division
import sys
import argparse
import numpy as np
import subprocess as sp
import os.path
import dtw
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy.signal.filter_design import butter, buttord
from scipy.signal import lfilter, lfiltic
from pydub import AudioSegment
from features import mfcc
from features import logfbank

FFMPEG_BIN = 'ffmpeg'

def chop(input_path, output_path, start_ms=30, end_ms=30):
  if check_file_paths([input_path]) == -1:
    return -1
  sound = AudioSegment.from_file(input_path, format="wav")
  chopped_sound = sound[start_ms:-end_ms]
  chopped_sound.export(output_path, format="wav")

def trim(input_path, output_path, silence_threshold=-40.0, chunk_size=10):
    if check_file_paths([input_path]) == -1:
        return -1
    sound = AudioSegment.from_file(input_path, format="wav")
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]
    trimmed_sound.export(output_path, format="wav")

def match_length(input_path, output_path, match_path, force=False):
    if check_file_paths([input_path, match_path]) == -1:
        return -1
    
    input_segment = AudioSegment.from_file(input_path)
    input_length = input_segment.duration_seconds
    match_segment = AudioSegment.from_file(match_path)
    match_seg_length = match_segment.duration_seconds

    length_coefficient = input_length / match_seg_length
    if length_coefficient < 2.0 and length_coefficient > 0.5:
        change_length(input_path, output_path, length_coefficient)
        return 0
    if force:
        if length_coefficient > 2.0:
            change_length(input_path, input_path, 2.0)
            match_length(input_path, output_path, match_path, force=True)
        else:
            change_length(input_path, input_path, 0.5)
            match_length(input_path, output_path, match_path, force=True)
    else:
        print 'wrong size'
        return -1

def filter(input_path, output_path):
    if check_file_paths([input_path]) == -1:
        return -1
    data, rate = ffmpeg_load_audio(input_path, 44100, True, dtype=np.float32)
    filtered_data = butter_bandpass_filter(data, 100.0, 3000.0, 44100)
    wavwrite(output_path, 44100, filtered_data)

def compare(control_path, exp_path):
    (rate,sig) = wavread(control_path)
    (rate2,sig2) = wavread(exp_path)

    x = mfcc(sig,rate)
    y = mfcc(sig2,rate2)

    dist, cost, acc = dtw.dtw(x, y, dist=lambda x, y: dtw.norm(x - y, ord=1))\

    return dist

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def ffmpeg_load_audio(filename, sr=44100, mono=True, dtype=np.float32):
    channels = 1 if mono else 2
    format_strings = {
        np.float64: 'f64le',
        np.float32: 'f32le',
        np.int32: 's32le',
        np.uint32: 'u32le'
    }
    format_string = format_strings[dtype]
    command = [
        'ffmpeg',
        '-i', filename,
        '-f', format_string,
        '-acodec', 'pcm_' + format_string,
        '-ar', str(sr),
        '-ac', str(channels),
        '-loglevel', 'quiet',
        '-']
    p = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    bytes_per_sample = np.dtype(dtype).itemsize
    chunk_size = bytes_per_sample * channels * sr # read in 1-second chunks
    raw = b''
    with p.stdout as stdout:
        while True:
            data = stdout.read(chunk_size)
            if data:
                raw += data
            else:
                break
    audio = np.fromstring(raw, dtype=dtype)
    if channels > 1:
        audio = audio.reshape((-1, channels)).transpose()
    return(audio, sr)


def change_length(input_path, output_path, length_coefficient, log=False):

    atempo = 'atempo=' + str(length_coefficient)

    command = [ FFMPEG_BIN,
              '-i', input_path,
              '-filter:a',
              atempo,
              '-acodec', 'pcm_f32le',
              '-y']
    if not log:
        command.append('-loglevel')
        command.append('8')
    command.append(output_path)
    sp.call(command, shell=False)

def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    trim_ms = 0 # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms

def check_file_paths(file_path_list):
    for file_path in file_path_list:
        if not os.path.exists(file_path):
            print('File: ' + file_path + ' does not exist.')
            return -1
