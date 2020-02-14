import pickle
import pandas as pd
import speech_recognition as sr

# 音轉字
test_csv = pd.read_csv('./submission.csv')
y = test_csv['label'].values
filenames = test_csv['wav'].tolist()

test_rowdata = []
for filename in filenames:
    r = sr.Recognizer()
    voice = sr.AudioFile("./voice_dataset/test_set/{}".format(filename))
    with voice as source:
        r.adjust_for_ambient_noise(source) #降噪
        audio = r.record(source)

    s = r.recognize_google(audio,language='zh-CHS')
    test_rowdata.append(s)

with open('test_rowdata.pickle','wb') as file:
    pickle.dump(test_rowdata,file)
