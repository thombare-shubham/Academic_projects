import numpy as np

# SPLIT AUDIO
def split_Audio(audio_data, w, h, threshold_level, tolerance=10):
    split_map = []
    start = 0
    data = np.abs(audio_data)  # To get the absolute value
    # np.mean is used to compute arithmetic mean of threshold level
    threshold = threshold_level*np.mean(data[:25000])
    inside_sound = False  # To ensure that there is no audio playback
    near = 0

    for i in range(0, len(data)-w, h):
        win_mean = np.mean(data[i:i+w])  # mean of window length
        # if sound is having higher threshold than average threshold of audio file and its playable
        if(win_mean > threshold and not (inside_sound)):
            inside_sound = True  # playable audio-clip
            start = i  # reset the audio starting point
        if(win_mean <= threshold and inside_sound and near > tolerance):
            inside_sound = False
            near = 0
            split_map.append([start, i])  # Insert trimmed audio file
        if(inside_sound and win_mean <= threshold):
            near += 1
    return split_map  # return trimmed audio file
