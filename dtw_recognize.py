"""
CS 73 Computational Linguistics
Assignment 6 - Speech Recognition
Authors: Alex Gerstein, Scott Gladstone

We will use a portion of the ISOLET corpus, 
consisting of recordings of isolated letters (A-Z) 
by different speakers with different accents. 
The training portion is a set of 12 recordings 
for each word (where a word is one of the letters 
A-Z). MFCCs have been pre-computed over frames of 
25 ms wide, every 10 ms. Each row in a .mfc file 
represents the cepstral coefficients from one time 
frame. The filenames are of the form "speaker-letter1-t.mfc".

This program implements a procedure to infer 
which letter is being spoken in unseen speech 
recording, by using k-nearest-neighbors to match 
the input speech recording against each training 
recording, and hypothesizing the majority label 
of the k nearest training examples. The distance 
between pairs of recordings is computed with DTW.

The program reads the training and testing MFCC files,
recognizes which letter is being said by each 
of the testing files using kNN (with k=3) and 
dynamic time warping, and prints an accuracy 
score for the recognition.

Goal: accuracy = 73.08%
"""

import argparse
import glob
from scipy.spatial import distance
from collections import Counter

MFC_TAG = ".mfc"
KNN = 3


class SpeechRecognizer:

    def __init__(self, train_folder, test_folder):
        self.train_set = self._vectorize_data(train_folder)
        self.test_set = self._vectorize_data(test_folder)

    def train_data(self):
        """
        Trains vectorized data using a kNN classifier with majority
        voting and Dynamic Time Warping (DTW) for feature distances
        """
        data_labels = {}
        for test_key, test_vector in self.test_set.iteritems():
            print "Training on:", test_key
            distances = []
            
            for train_key, train_vector in self.train_set.iteritems():
                dtw = self._dtw(test_vector, train_vector)
                distances.append((train_key, dtw))

            data_labels[test_key] = self._get_majority(distances)
        
        return data_labels

    def test_data(self, labels):
        count = 0
        for key, match in labels.iteritems():
            count += (match == key.split('-')[1].strip("0123456789"))
        print labels
        return 1.0 * count / len(labels)

    def _vectorize_data(self, folder):
        """
        Converts data text files in matrices of feature vectors 
        index-able in a dictionary by text filename
        """
        data_set = {}
        for mfc_file in glob.glob(folder + "/*" + MFC_TAG):
            with open(mfc_file, 'r') as mf:
                data_set[mfc_file] = []
                for line in mf:
                    line = map(float, line.split())
                    data_set[mfc_file].append(line)
        return data_set

    def _dtw(self, feat_test, feat_train):
        """
        Dynamic Time Warping (DTW): edit distance between speech feature vectors
        """
        m = len(feat_test) + 1      # rows in distance matrix
        n = len(feat_train) + 1     # cols in dist matrix
        
        dtw = [[0.0] * n for i in range(m)]

        # Initialize first row and column to all infinities
        for i in range(1, m):
            dtw[i][0] = float("inf")
        for j in range(1, n):
            dtw[0][j] = float("inf")

        # Find minimum cost path 
        # s.t. D[i][j] = cost(i,j) + min(upper-left, up, left)
        costs = distance.cdist(feat_test, feat_train)   # matrix of euclidean dists
        for i in range(1, m):
            for j in range(1, n):
                dtw[i][j] = costs[i-1][j-1] + min(dtw[i-1][j-1], dtw[i-1][j], dtw[i][j-1])
        return dtw[-1][-1]

    def _get_majority(self, distances):
        """
        Gets majority vote by label. Chooses smallest value if no majority.
        """
        sorted_dists = sorted(distances, key=lambda x: x[1])
        labels = [dist[0] for dist in sorted_dists[:KNN]]
        labels = map(lambda x: x.split('-')[1].strip("0123456789"), labels)
        
        if labels[0] != labels[1] != labels[2] != labels[0]:
            majority = labels[0]
        else:
            majority = Counter(labels).most_common(1)[0][0]

        return majority

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A speech recognition program using dynamic time warping.")
    parser.add_argument('folder', help="A folder that contains testing and training subfolders of MFC files.")
    args = parser.parse_args()

    training_folder = args.folder + '/train'
    testing_folder = args.folder + '/test'

    sr = SpeechRecognizer(training_folder, testing_folder)
    learned_labels = sr.train_data()
    accuracy = sr.test_data(learned_labels)
    print "The accuracy on the data is", '{:.2%}'.format(accuracy)