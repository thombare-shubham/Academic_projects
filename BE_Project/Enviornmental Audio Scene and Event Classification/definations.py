from sklearn.preprocessing import LabelEncoder
from tkinter import *

global activities
global subjects
global train_subjects
global validation_subjects
global test_subjects
global chars
global charslen
global lb

main_ui = Tk()

# 32 Events
activities = ['CanOpening', 'CarHorn', 'Cat', 'ChirpingBirds', 'Clapping', 'ClockAlarm', 'Cow', 'CracklingFire', 'Crow', 'CryingBaby', 'Dog', 'Door_or_WoodCreaks', 'Engine', 'Fireworks', 'GlassBreaking', 'HandSaw', 'Helicopter', 'Hen', 'Laughing', 'Night', 'Pig', 'Rain', 'Rooster', 'SeaWaves', 'Siren', 'Snoring', 'Thunderstorm', 'Train', 'TwoWheeler', 'VaccumCleaner', 'WaterDrops', 'Wind']

subjects = ['s01', 's02', 's03', 's04', 's05']

train_subjects = ['s01', 's02','s03']
validation_subjects = ['s04']
test_subjects = ['s05']

chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-" 
charsLen = len(chars)

# Replicate LableEncoder
# LabelEncoder is used for normalizing label values
lb = LabelEncoder()