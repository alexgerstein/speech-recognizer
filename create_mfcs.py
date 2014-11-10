import argparse
import glob
from scipy.io import wavfile
import numpy as np

from scipy.signal import lfilter, hamming
from scipy.fftpack import fft
from scipy.fftpack.realtransforms import dct

from scikits.talkbox import segment_axis
from scikits.talkbox.features.mfcc import trfbank, preemp

AUDIO_TAG = ".wav"
MFCC_TAG = ".mfc"


def mfcc(input, nwin=0.025, nfft=512, fs=16000, overlap=0.01, nceps=13):  #added overlap as a parameter
    """Compute Mel Frequency Cepstral Coefficients.

    Parameters
    ----------
    input: ndarray
        input from which the coefficients are computed

    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.

    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum

    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""

    overlap = int(overlap * fs)
    nwin = int(nwin * fs)   #added these two lines to convert from time to frames

    over = nwin - overlap #changed from nwin - 160


    # MFCC parameters: taken from auditory toolbox

    # Pre-emphasis factor (to take into account the -6dB/octave rolloff of the
    # radiation at the lips level)
    prefac = 0.97

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    nfil = nlinfil + nlogfil

    w = hamming(nwin, sym=0)

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]

    #------------------
    # Compute the MFCC
    #------------------
    extract = preemp(input, prefac)
    framed = segment_axis(extract, nwin, over) * w

    # Compute the spectrum magnitude
    spec = np.abs(fft(framed, nfft, axis=-1))
    # Filter the spectrum through the triangle filterbank
    mspec = np.log10(np.dot(spec, fbank.T))
    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]

    return ceps, mspec, spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A speech recognition program using dynamic time warping.")
    parser.add_argument('folder', help="A folder that contains testing and training subfolders of MFC files.")
    args = parser.parse_args()

    training_folder = args.folder + '/train'
    testing_folder = args.folder + '/test'

    for training_file in glob.glob(training_folder + "/*" + AUDIO_TAG):
        print training_file
        fs, data = wavfile.read(training_file)
        ceps, mspec, spec = mfcc(data, fs=fs)

        output_filename = training_file.replace(AUDIO_TAG, MFCC_TAG)
        with open(output_filename, 'w') as outfile:
            for vector in ceps:
                outfile.write(" ".join(map(str, vector)) + "\n")

    for testing_file in glob.glob(testing_folder + "/*" + AUDIO_TAG):
        print testing_file
        fs, data = wavfile.read(testing_file)
        ceps, mspec, spec = mfcc(data, fs=fs)

        output_filename = testing_file.replace(AUDIO_TAG, MFCC_TAG)
        with open(output_filename, 'w') as outfile:
            for vector in ceps:
                outfile.write(" ".join(map(str, vector)) + "\n")