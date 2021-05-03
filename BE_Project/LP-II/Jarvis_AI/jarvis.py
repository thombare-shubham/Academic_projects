import pyttsx3
import speech_recognition as sr
import datetime
# import wikipedia
import webbrowser
import os
import smtplib
from pymongo import MongoClient #To be used for database work

# Initializing database client
client = MongoClient('localhost', 27017)
print(client.server_info())#Prints server info

#  Creating New databse for VA
voice_database = client['jarvis']

# Collection in database
col_jarvis = voice_database["command_details"]

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[0].id)

# Set Voice property
engine.setProperty('voice',voices[0].id)

# Function to let jarvis speak
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# Function for wishing
def wishMe():
        hour = int(datetime.datetime.now().hour)
        if hour>=0 and hour<=12:
            speak("Good Morning!")
        elif hour>=12 and hour<=17:
            speak("Good Afternoon!")
        else:
            speak("Good Evening!")
        speak("I am jarvis, your personal assistant, Please tell me how may I help you")

# takeCommand function for taking command as a voice from user 
'''basically takes input from microphone and converts it in text'''

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        r.energy_threshold = 1000
        r.pause_threshold = 0.5
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio) #Language='en-US' is removed
        print(f"User Said: {query}\n")

    except Exception as e:
        print(e)

        print("Say that again please...")
        return "None"
    return query

# Send Email function
def sendEmail(to, content):
    document = {"voice":voices[0].id,'command':"send email", 'date & time': datetime.datetime.now(),"status":"success"}
    col_jarvis.insert_one(document)
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.ehlo()
    server.starttls()
    server.login('Your_Email@gmail.com','Password')
    server.sendmail('Your_Email@gmail.com', to, content)
    server.close

# Main Function
if __name__ == "__main__":
    wishMe()
    while True:
    # if 1:
        query = takeCommand().lower()

        # Logic to execute a task based on query
        if 'search' in query:
            try:
                speak('searching ...')
                query = query.replace("search","")
                results = webbrowser.open(query)
                # speak("According to google")
                print(results)
                speak(results)
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'open youtube' in query:
            try:
                webbrowser.open("youtube.com")
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'open google' in query:
            try:
                webbrowser.open("google.com")
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'open gmail' in query:
            try:
                webbrowser.open("gmail.com")
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'open stack overflow' in query:
            try:
                webbrowser.open("stackoverflow.com")
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'play music' in query:
            try:
                music_dir = 'C:\\Users\\shubh\\Music\\English'
                songs= os.listdir(music_dir)
                print(songs)
                os.startfile(os.path.join(music_dir, songs[0]))
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif "what's the time" in query:
            try:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")
                speak(f"Sir, the time is {strTime}")
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'open code' in query:
            try:
                codePath = "C:\\Users\\shubh\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
                os.startfile(codePath)
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'send mail to shubham' in query:
            try:
                speak("what should I say?")
                content = takeCommand()
                to = "shubhamrthombare2@gmail.com"
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                print(e)
                speak("Sorry Email was not sent!")
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'give me database details' in query:
            try:
                print(f"\nDatabases")
                print(client.list_database_names())
                print(f"\nCollections in the Database")
                print(voice_database.list_collection_names())
                print(f"\nDocuments in the Collection")
                dessert = col_jarvis.find()
                # Print each Document
                for des in dessert:
                    print(des)
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        elif 'thank you' in query:
            try:
                speak('Your welcome, I would like to help u again,Good Bye!')
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"success"}
                col_jarvis.insert_one(document)
                exit()
            except Exception as e:
                document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
                col_jarvis.insert_one(document)

        else:
            document = {"voice":voices[0].id,'command':query, 'date & time': datetime.datetime.now(),"status":"failed"}
            col_jarvis.insert_one(document)