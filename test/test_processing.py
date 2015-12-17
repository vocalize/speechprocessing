
import processing
import os

def length(path):

    f=open(path,"r")

    #read the ByteRate field from file (see the Microsoft RIFF WAVE file format)
    #https://ccrma.stanford.edu/courses/422/projects/WaveFormat/
    #ByteRate is located at the first 28th byte
    f.seek(28)
    a=f.read(4)

    #convert string a into integer/longint value
    #a is little endian, so proper conversion is required
    byteRate=0
    for i in range(4):
        byteRate=byteRate + ord(a[i])*pow(256,i)

    #get the file size in bytes
    fileSize=os.path.getsize(path)  

    ms=((fileSize-44)*1000)/byteRate
    return ms

def test_compare():

    assert int(processing.compare('compare/umbrella.wav', 'compare/westernUmbrella.wav')) == 111

def test_butterBandpassFilter():

    data, rate = processing.ffmpeg_load_audio('butterBandpassFilter/umbrella.wav', 44100, True, dtype=processing.np.float32)
    output = processing.np.loadtxt('butterBandpassFilter/test.txt')
    assert all([i==j for i,j in zip(processing.butter_bandpass_filter(data, 100.0, 3000.0, 44100), output)])

def test_detectLeadingSilence():

    sound = processing.AudioSegment.from_file('detectSilence/leading/umbrella.wav', format="wav")
    assert processing.detect_leading_silence(sound) == 20

def test_detectEndingSilence():

    sound = processing.AudioSegment.from_file('detectSilence/Ending/umbrella.wav', format="wav")
    assert processing.detect_leading_silence(sound.reverse()) == 20

def test_ffmpegload():
    
    out = processing.ffmpeg_load_audio('ffmpegload/output.wav')
    assert any([i==j for i,j in zip(processing.ffmpeg_load_audio('ffmpegload/input.wav'), out)])

def test_match():

    assert processing.match_length('match/input/73energy.wav', 'match/correctedInput', 'match/test/energy.wav') == 0

def test_matchShort():

    assert processing.match_length('match/input/44energyShort.wav', 'match/correctedInput', 'match/test/energy.wav') == -1

def test_matchLong():

    assert processing.match_length('match/input/64energyLong.wav', 'match/correctedInput', 'match/test/energy.wav') == -1

def test_changeLength():

    processing.change_length('changeLength/73energy.wav', 'changeLength/73energyShort.wav', 2.0)
    duration = length('changeLength/73energyShort.wav')

    assert duration == 251

def test_chop():

    processing.chop('chop/umbrella.wav', 'chop/umbrellachopped.wav')
    duration = length('chop/umbrellachopped.wav')

    assert duration == 597

def test_trim():

    processing.trim('trim/umbrella.wav', 'trim/umbrellatrimmed.wav')
    duration = length('trim/umbrellatrimmed.wav')

    assert duration == 637

def test_filter():
    
    processing.filter('filter/umbrella16000.wav', 'filter/umbrella44100.wav')
    fs = processing.wavread('filter/umbrella44100.wav')
    
    assert fs[0] == 44100

def test_average():

    processing.average('average/output/averaged.wav', 'average/input/19energy.wav', 'average/input/44energy.wav', 'average/input/64energy.wav')
    sampFreq, snd = processing.wavread('average/output/averaged.wav')
    testResult = processing.np.loadtxt('trim/test.txt')
    
    assert all([i==j for i,j in zip(snd, testResult )])

   
   





