import numpy as np
import scipy as sp
import os
import librosa, librosa.display
import math
import json

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    # build dictionary to store data
    data = {
        "mapping" : [ ],
        "mfcc" : [],
        "labels" : []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all the genres
    # os.walk: 어떤 경로의 모든 하위 폴더와 파일을 탐색

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we are not at the root level

        if dirpath is not dataset_path: # dataset_path 아래 하위 항목이 있을 경우와 같다.
            semantic_label = dirpath.split("/")[-1] # genre/blues -> ["genre", "blues"]
            data["mapping"].append(semantic_label)

            print("\nProcessing: {}".format(semantic_label))

            for f in filenames:
                # load audio file
                try:
                    file_path = os.path.join(dirpath, f) #dirpath 내 파일 경로 중 파일 f의 path를 참조한다
                    signal, sample_rate = librosa.load(file_path, sr = SAMPLE_RATE)

                    # process segments extracting mfcc and storing data
                    for d in range(num_segments):

                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_fft = n_fft, n_mfcc = n_mfcc, 
                                                    hop_length = hop_length)
                        mfcc = mfcc.T

                        
                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)

                            print("{}, segments:{}".format(file_path, d+1))
                except:
                    print("file open failed")
                    pass
    with open(json_path, "w+") as fp:
        json.dump(data, fp, indent = 4)