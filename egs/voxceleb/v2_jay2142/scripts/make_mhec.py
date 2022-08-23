from scipy.io import wavfile
from scipy.signal import lfilter, gammatone, hilbert
from scipy.fft import dct
import numpy as np
from librosa.util import frame
import sys, os
from kaldiio import ReadHelper, WriteHelper


def preprocess(signal, sample_rate, alpha=0.95):
    """
    pre-process a wav file
    alpha determines the pre-emphasis
    """
    signal = np.squeeze(signal)

    if alpha is not None and alpha != 0:
        signal = lfilter([1., -alpha],1,signal)
    signal = signal / 32768 # This is a common hack to normalize 16-bit PCM WAV data to the [-1, 1] range!
    return signal

def construct_gammatone_filters(sample_rate=16000):
    """
    I create 10 gammatone filters with center frequencies ranging from 200 to 1100 inclusive.
    """
    gammatone_filters = []
    filter_order = 4
    filter_length = 2048
    for center_frequency in range(200, 1200, 100): # full range is 3400
        gammatone_filters.append(gammatone(freq=center_frequency, ftype='fir', order=filter_order, numtaps=filter_length, fs = sample_rate))
    return gammatone_filters

def convert_to_gammatone(signal, sample_rate = 16000):
    """
    I apply the 10 gammatone filters, creating a 10-channel signal. 
    The hilbert envelope is computed for each gammatone filter.
    """
    filters = construct_gammatone_filters(sample_rate=sample_rate)
    filter_outputs = []
    envelopes = []
    for filter in filters:
        filter_output = lfilter(filter[0], filter[1], signal)
        filter_outputs.append(filter_output)
        envelope = np.abs(hilbert(filter_output))
        envelopes.append(envelope)
    return np.array(envelopes)

def smoothen_envelopes(envelopes, f_c = 20, sample_rate = 16000):
    '''
    I smooth the envelopes following the equation in the MHEC paper
    f_c is cut-off frequency, which is 20 as defined in the paper
    '''
    eta = np.exp(-2 * np.pi * f_c / sample_rate)
    smooth_envelopes = np.zeros(envelopes.shape)
    smooth_envelopes[:, 0] = (1 - eta) * envelopes[:, 0]
    for i in range(1, envelopes.shape[1]):
        smooth_envelopes[:, i] = (1 - eta) * envelopes[:, i] + eta * smooth_envelopes[:, i-1]
    return smooth_envelopes
def create_frames(envelopes, frame_length_ms=0.025, skip_length_ms = 0.010, sample_rate = 16000):
    """
    I split the MHEC data into Hamming-windowed frames, with frame and skip lengths as dictated in the MHEC paper.
    """
    frame_length = int(sample_rate * frame_length_ms)
    skip_length = int(sample_rate * skip_length_ms)
    frames = frame(envelopes, frame_length=frame_length, hop_length=skip_length, axis=1)
    hamming_window = np.hamming(frame_length)
    hamming_frames = frames * hamming_window
    return hamming_frames

def compute_sample_means(hamming_frames):
    """
    I compute the sample mean of each frame
    """
    return np.mean(hamming_frames, axis=2)



def compute_mhec(signal, sample_rate):
    """
    Calls all the above functions in the appropriate sequence
    """
    preprocessed_signal = preprocess(signal=signal, sample_rate=sample_rate, alpha=0)
    envelopes = convert_to_gammatone(preprocessed_signal, sample_rate)
    smooth_envelopes = smoothen_envelopes(envelopes)
    hamming_frames = create_frames(smooth_envelopes)
    means = compute_sample_means(hamming_frames)
    log_means = np.log(means)
    mhec = np.swapaxes(dct(log_means), 0, 1)
    return mhec

def compute_mhec_from_file(wav_path):
    sample_rate, signal = wavfile.read(wav_path)
    signal = signal / 32768
    return compute_mhec(signal, sample_rate)

def main():
    if len(sys.argv) != 3:
        print("Need 2 arguments: a wav rspecifier and a feats wspecifier")
        return
    with ReadHelper(sys.argv[1]) as reader, WriteHelper(sys.argv[2]) as writer:
        print(os.path.abspath(sys.argv[1]))
        print(os.getcwd())
        for key, (rate, wav_data) in reader:
            signal = preprocess(wav_data, rate, alpha=0)
            mhec = compute_mhec(signal, rate)
            mhec = np.pad(mhec, (1, 1), 'constant')
            writer(key, mhec)
if __name__ == '__main__':
    main()
