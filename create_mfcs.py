import argparse
import glob
from scipy.io import wavfile
import scikits.talkbox.features

AUDIO_TAG = ".wav"
MFCC_TAG = ".mfc"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A speech recognition program using dynamic time warping.")
    parser.add_argument('folder', help="A folder that contains testing and training subfolders of MFC files.")
    args = parser.parse_args()

    training_folder = args.folder + '/train'
    testing_folder = args.folder + '/test'

    for training_file in glob.glob(training_folder + "/*" + AUDIO_TAG):
        print training_file
        fs, data = wavfile.read(training_file)
        mfcc_data = scikits.talkbox.features.mfcc(data, fs=fs, nwin=0.025 * fs)[0]

        output_filename = training_file.replace(AUDIO_TAG, MFCC_TAG)
        with open(output_filename, 'w') as outfile:
            for vector in mfcc_data:
                for dist in vector:
                    outfile.write(str(dist) + " ")

                outfile.write("\n")
