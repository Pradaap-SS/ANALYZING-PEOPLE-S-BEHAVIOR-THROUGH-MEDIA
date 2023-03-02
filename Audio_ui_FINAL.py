from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os
from tkinter import messagebox
from PIL import Image, ImageTk

import paralleldots
import json
import os
import pandas as pd
import librosa
import glob
import librosa
import librosa.display
import pydub
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

from pydub import AudioSegment
import moviepy.editor

window=Tk()
window.geometry("700x700")
window.title("Emotion Analyzer")

aud=StringVar()
out=StringVar()
final_out=StringVar()

def browsefunc():
    filename=filedialog.askopenfilename()
    pathlabel.config(text=os.path.basename(filename))
    print(filename)
    #aud.set(os.path.basename(filename))
    if(messagebox.askokcancel("Submit",filename)):
        aud.set(str(filename))

def Audio_features():
    
    media_file=aud.get()
    import subprocess
    subprocess.call(['ffmpeg', '-i', media_file ,'audio.wav'])
    
    import speech_recognition as sr
    AUDIO_FILE = 'audio.wav'
    r = sr.Recognizer() 
  
    with sr.AudioFile(AUDIO_FILE) as source: 
    #reads the audio file. Here we use record instead of 
    #listen 
        audio = r.record(source)   
  
    try:
        r.recognize_google(audio)
        #print("The audio file contains: " + r.recognize_google(audio)) 
  
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio") 
  
    except sr.RequestError as e: 
        print("Could not request results from Google Speech Recognition service; {0}".format(e)) 


    api_key = "ANxhDZ8aioVsDqhRDMkqgNgkNRpzjxVdqGFPhBPtYrQ"
    paralleldots.set_api_key( api_key )
    text=r.recognize_google(audio)
    
    print( "\nEmotion" )
    print( paralleldots.emotion(text))
    y=json.dumps(paralleldots.emotion(text))
    z=json.loads(y)
    print(z['emotion'])
    final_out.set(z['emotion'])
    
    #waveform
    #AUDIO_FILE = aud.get()
    data, sampling_rate = librosa.load(AUDIO_FILE)
    plt.figure(figsize=(4, 2))
    librosa.display.waveplot(data, sr=sampling_rate)
    plt.show()
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig=plt.figure(figsize=(4, 2))
    canvas=FigureCanvas(fig)
    ax=fig.add_subplot(111)
    p=librosa.display.waveplot(data, sr=sampling_rate)
    fig.savefig('Wave.png')
    
    canvas= Canvas(tab2)
    #canvas.pack()
    img=PhotoImage(file="Wave.png")
    canvas.create_image(0,0,anchor=NW,image=img)
    canvas.image=img
    canvas.place(x=200,y=100)

    #spectogram
    y, sr = librosa.load(AUDIO_FILE)
    y = y[:100000] # shorten audio a bit for speed

    window_size = 1024
    window = np.hanning(window_size)
    stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
    out = 2 * np.abs(stft) / np.sum(window)

    # For plotting headlessly
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig1= plt.Figure(figsize=(4, 2))
    canvas = FigureCanvas(fig1)
    ax = fig1.add_subplot(111)
    p =librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
    fig1.savefig('spec.png')
    
    canvas1= Canvas(tab2)
    #canvas1.pack()
    img1=PhotoImage(file="spec.png")
    canvas1.create_image(0,0,anchor=NW,image=img1)
    canvas1.image=img1
    canvas1.place(x=200,y=250)

      
    # use the audio file as the audio source 
    

    
#notebook creation
window.title("Emotion Analyzer")
tab_control = ttk.Notebook(window)
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text='Audio')
tab_control.pack(expand=1, fill='both')



#audio module
A_label0=Label(tab2,text="Upload Audio File: ",width=20,font=("arial",10,"bold"))
A_label0.place(x=20,y=25)

file_loc= Label(tab2,textvariable=aud,font=("arial",10,"bold"),bg="white",height=1,width=40)
file_loc.place(x=200,y=25)

A_browse =Button(tab2, text='Browse', width=10, command=browsefunc) #command = some function
A_browse.place(x=200,y=50)

A_submit=Button(tab2, text='Submit', width=10,command=Audio_features) #command = some function
A_submit.place(x=300,y=50)


pathlabel= Label(window)
pathlabel.pack()

A_label1=Label(tab2,text="Audio Features  ",width=20,font=("arial",10,"bold"))
A_label1.place(x=20,y=100)

A_label2=Label(tab2,text="Waveform : ",width=20,font=("arial",10,"bold"))
A_label2.place(x=20,y=150)

A_label3=Label(tab2,text="Spectrogram : ",width=20,font=("arial",10,"bold"))
A_label3.place(x=20,y=300)

A_label4=Label(tab2,text="Final Emotion Detected: ",width=20,font=("arial",10,"bold"))
A_label4.place(x=20,y=550)

A_label5= Label(tab2,textvariable=final_out,font=("arial",10,"bold"),bg="white",height=2,width=40)
A_label5.place(x=200,y=550)



window.mainloop()
