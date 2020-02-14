import pickle
import pandas as pd
import speech_recognition as sr


## 音轉字
train_csv = pd.read_csv('./train.csv')
y = train_csv['label'].values
filenames = train_csv['wav'].tolist()

train_rowdata = []
for filename in filenames:
    r = sr.Recognizer()
    voice = sr.AudioFile("./voice_dataset/train_set/{}".format(filename))
    with voice as source:
        r.adjust_for_ambient_noise(source) #降噪
        audio = r.record(source)

    s = r.recognize_google(audio,language='zh-CHS')
    print(s)
    train_rowdata.append(s)

with open('train_rowdata.pickle','wb') as file:
    pickle.dump(train_rowdata,file)

