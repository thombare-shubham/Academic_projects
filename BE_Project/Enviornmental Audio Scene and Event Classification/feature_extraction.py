import os
from tkinter import messagebox
# Import files
from save_stft import *
from definations import *

# SAVE FEATURES
def Feature_Extraction():
    for activity in activities:
        for subject in subjects:
            innerDir = subject + "/" + activity
            for file in os.listdir("Dataset_audio/"+innerDir+"/"):
                if(file.endswith(".wav")):
                    save_STFT("Dataset_audio/"+innerDir+"/" + file, file, activity, subject)
                    print("Extracting feature from "+subject+"-"+file+"-"+activity)

    messagebox.showinfo("Features Extracted","Features of audio files are extracted!")