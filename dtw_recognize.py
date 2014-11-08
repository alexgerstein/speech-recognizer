"""
We will use a portion of the ISOLET corpus, 
consisting of recordings of isolated letters (A-Z) 
by different speakers with different accents. 
The training portion is a set of 12 recordings 
for each word (where a word is one of the letters 
A-Z). MFCCs have been pre-computed over frames of 
25 ms wide, every 10 ms. Each row in a .mfc file 
represents the cepstral coefficients from one time 
frame. The filenames are of the form "speaker-letter1-t.mfc".

Your task is to implement a procedure to infer 
which letter is being spoken in unseen speech 
recording, by using k-nearest-neighbors to match 
the input speech recording against each training 
recording, and hypothesizing the majority label 
of the k nearest training examples. The distance 
between pairs of recordings is computed with DTW.

Write a program named dtw_recognize.py that reads
the training and testing MFCC files,
recognizes which letter is being said by each 
of the testing files using kNN (with k=3) and 
dynamic time warping, and prints an accuracy 
score for the recognition.
"""

import argparse


def train_data(folder):
    return


def test_data(folder):
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A speech recognition program using dynamic time warping.")
    parser.add_argument('folder', help="A folder that contains testing and training subfolders of MFC files.")
    args = parser.parse_args()

    training_folder = args.folder + '/train'
    testing_folder = args.folder + '/test'

    train_data(training_folder)
    accuracy = test_data(testing_folder)
    print "The accuracy on the isolet data is", '{:.2%}'.format(accuracy)