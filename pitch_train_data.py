# coding: utf-8
"""
__author__ = 'BCking'
"""
import pitch_analyse
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.model_selection import train_test_split


def creat_series(pitchs, label):
    pitch_list = []
    time_list = []
    id_list = []
    for pitch in pitchs:
        pitch_list.extend(pitch)
        time_list.extend(range(len(pitch)))
        id_list.extend([label]*len(pitch))
        label += 1
    return (pitch_list, time_list, id_list, label)
    
    
def get_series():
    emotions = ['angry',  'fear', 'happy', 'normal', 'sad', 'surprise']
    pitch_list = []
    time_list = []
    emotion_idx = []
    id_list = []        
    label = 0           
    y = []              
    for idx, emotion in enumerate(emotions):
        emotion_len = 0
        emotion_pitch = []
        for line in open("pitch_data/"+emotion+"_pitch_deviation.txt"):
            line = line.strip()
            if not line:
                continue
            line = np.array(line.split(), dtype=np.float32)
            emotion_pitch.append(line)
            y.append(idx)
            emotion_len += 1
        emotion_idx.append(emotion_len)
        ll = creat_series(emotion_pitch, label)
        pitch_list.extend(ll[0])
        time_list.extend(ll[1])
        id_list.extend(ll[2])
        label = ll[3]
    series = {'id': id_list, 'time': time_list,'value': pitch_list}
    y = pd.Series(y)
    series = pd.DataFrame(series)
    return series, y, emotion_idx

    
def feature_extract(series):
    extracted_features = extract_features(series, column_id="id",  column_sort='time')
    return extracted_features
    

def feature_select(extracted_features, y):
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    return features_filtered

    
def data_writer(features_filtered, emotion_idx):
    emotions =  ['angry',  'fear', 'happy', 'normal', 'sad', 'surprise']
    start_idx = 0
    for idx, emotion in enumerate(emotions):
        fw_train = open("pitch_data/"+emotion+"_train_d.em", 'w+')
        fw_test = open("pitch_data/"+emotion+"_test_d.em", 'w+')
        emotion_len = emotion_idx[idx]
        emotion_feature = features_filtered[start_idx: start_idx+emotion_len]
        np.random.shuffle(emotion_feature)
        print idx, emotion_len, len(features_filtered), len(emotion_feature)
        for i in range(0, emotion_len/4*3):
            fw_train.write(' '.join([str(j) for j in emotion_feature[i]]) + '\n')
        for i in range(emotion_len/4*3, emotion_len):
            fw_test.write(' '.join([str(j) for j in emotion_feature[i]]) + '\n')
        fw_test.close()
        fw_train.close()
        start_idx += emotion_len

    
if __name__ == "__main__":
    series, y, emotion_idx = get_series()
    print series, y, emotion_idx
    # from tsfresh.examples import load_robot_execution_failures
    #
    # series, y = load_robot_execution_failures()
    # series = series.drop(['b', 'c','d','e','f'],axis=1)

    # y_train, y_test = train_test_split(y)
    # x_train = pd.DataFrame(index=y_train.index)
    # x_test = pd.DataFrame(index=y_test.index)
    # series_train = series[series['id'].isin(y_train.index)]
    # series_test = series[series['id'].isin(y_test.index)]
    # # x_train.reset_index(inplace=True)
    # print series_train, len(y_train)
    # print series_test, len(y_test)
    #

    # from tsfresh.transformers import RelevantFeatureAugmenter
    # augmenter = RelevantFeatureAugmenter(column_id="id", column_sort='time')
    # augmenter.set_timeseries_container(series_train)
    # augmenter.fit(x_train, y_train)
    # train_features = augmenter.transform(x_train)
    # print list(train_features)
    # augmenter.set_timeseries_container(series_test)
    # test_features = augmenter.transform(x_test)
    # print list(test_features)
    # print test_features

    extracted_features = feature_extract(series)
    print extracted_features
    features_filtered = feature_select(series, y)
    print features_filtered
    features_filtered = features_filtered.as_matrix()
    # data_writer(features_filtered, emotion_idx)

    
    
    