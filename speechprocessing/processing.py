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
    """
    Chops off a few milliseconds from the beginning and the end of an audio file.

    :param input_path: the input wav path
    :param output_path: the output wav path
    :param start_ms: the number of milliseconds to chop off the beginning of the file
    :param end_ms: the number of milliseconds to chop off the end of the file
    :returns: -1 if a file does not exist
    """
    if check_file_paths([input_path]) == -1:
        return -1
    sound = AudioSegment.from_file(input_path, format="wav")
    chopped_sound = sound[start_ms:-end_ms]
    # return chopped_sound
    chopped_sound.export(output_path, format="wav")


def trim(input_path, output_path, silence_threshold=-40.0, chunk_size=10):
    """
    Removes silence audio from the beginning and end of a wav file.

    :param input_path: the input wav path
    :param output_path: the output wav path
    :param silence_threshold: the volume that the function considers silence
    :param chunk_size: the step length in millsends (increase for a speed)
    :returns: -1 if a file does not exist
    """
    if check_file_paths([input_path]) == -1:
        return -1
    sound = AudioSegment.from_file(input_path, format="wav")
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]
    # return trimmed_sound
    trimmed_sound.export(output_path, format="wav")

def match_length(input_path, output_path, match_path, force=False):
    """
    Speeds up or slows down a wav file so that the length matches the length of another wav file.

    :param input_path: the input wav path
    :param output_path: the output wav path
    :param match_path: the path of the wav to match the length of
    :param force: call recursively if the input_path and match_path lengths vary greatly (not in betwee 0.5 and 2.0)
    :returns: -1 if a file does not exist or ffmpeg fails
    """
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

def filter(input_path, output_path, lowcut=100.0, highcut=3000.0, rate=44100):
    """
    Filters a wav file to get rid of frequencies not present in human speech. Band pass filter.

    :param input_path: the input wav path
    :param output_path: the output wav path
    :param lowcut: the lowest frequency to accept
    :param highcut: the highest frequency to accept
    :param rate: the sampling frequency
    :type rate: Number
    :param force: call recursively if the input_path and match_path lengths vary greatly (not in betwee 0.5 and 2.0)
    :returns: -1 if a file does not exist
    """
    if check_file_paths([input_path]) == -1:
        return -1
    data, rate = ffmpeg_load_audio(input_path, 44100, True, dtype=np.float32)
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, rate)
    # return filtered_data
    wavwrite(output_path, 44100, filtered_data)

def compare(control_path, exp_path):
    """
    Compares two wav files and returns a score. Uses mel frequency ceptrum coefficients as well as dynamic time warping.

    :param control_path: the 'correct' wav - what you are comparing to
    :param exp_path: the unknown wav
    """
    (rate,sig) = wavread(control_path)
    (rate2,sig2) = wavread(exp_path)

    x = mfcc(sig,rate)
    y = mfcc(sig2,rate2)

    dist, cost, acc = dtw.dtw(x, y, dist=lambda x, y: dtw.norm(x - y, ord=1))\

    return dist

def average(*args):
    """
    Averages multiple wav files together. Accomplishes this by performing fast fourier transforms on the data, averaging those arrays, and then performing an inverse fast fourier transform.

    :param args: array of wav paths with the output being the first path and the rest being inputs
    :returns: -1 if it fails or if it cannot find the paths

    :Example:

    >>> import speechprocessing
    >>> speechprocessing.average('output.wav', 'input_one.wav', 'input_two.wav')
    """
    if len(args) < 2:
        print 'Invalid number of arguments'
        return -1

    output_path = args[0]
    input_paths = args[1:]
    processed_wav_data = []
    if check_file_paths(input_paths) == -1:
        return -1
    for path in input_paths:
        data, rate = ffmpeg_load_audio(path, 44100, True, dtype=np.float32)
        filtered_data = butter_bandpass_filter(data, 100.0, 3000.0, 44100)
        processed_wav_data.append(filtered_data)

    fft_data = []

    for data in processed_wav_data:
        fft_data.append(np.fft.rfft(data))

    # Adding a * before an array of arrays makes it zip array-wise
    # .. or something. Nobody really knows how or why this works
    zipped_data = zip(*fft_data)

    mean_data = map(np.mean, zipped_data)

    # Reverse real fft
    averaged = np.fft.irfft(mean_data)
    wavwrite(output_path, 44100.0, averaged)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Runs a Butterworth bandpass filter on sound data

    :param data: sound data to filter
    :param lowcut: the lowest frequency to accept
    :param highcut: the highest frequency to accept
    :param fs: sampling frequency
    :param order: the order of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def ffmpeg_load_audio(filename, sr=44100, mono=True, dtype=np.float32):
    """
    Loads a wav file using ffmpeg to read and create a stream. Required because sometimes wav read doesn't work and no one (probably just me) knows why.

    :param filename: file path to load
    :param sr: sampling frequency
    :param mono: True if only one channel
    :param dtype: the data type to load the data as
    """
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
    """
    Change the length of an audio file

    :param input_path: the input wav path
    :param output_path: the output wav path
    :param length_coefficient: the number to multiply the length by ffmpeg only accepts numbers between 0.5 and 2.0
    :param log: log output to stderr
    """
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
    """
    Detects the amount of silence at the begnning of a wav file

    :param sound: the sound data
    :param silence_threshold: the volume to consider silence
    :param chunk_size: the step length in millsends (increase for a speed)
    """
    trim_ms = 0 # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms

def check_file_paths(file_path_list):
    """
    Checks all file paths in an array to make sure they exist

    :param file_path_list: array of file paths to check
    :returns: -1 if one or more files does not exist
    """
    for file_path in file_path_list:
        if not os.path.exists(file_path):
            print('File: ' + file_path + ' does not exist.')
            return -1
