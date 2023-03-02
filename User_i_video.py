from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os
from tkinter import messagebox

from os import path
from pydub import AudioSegment

import moviepy.editor

from PIL import Image, ImageTk

import live_cam
import media_file

import paralleldots
import json
import os
import pandas as pd
import librosa
import glob
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram




window=Tk()
window.geometry("700x700")
window.title("Emotion Analyzer")


aud=StringVar()
out=StringVar()
final_out=StringVar()
text_out=StringVar()
aud_out=StringVar()

def browsefunc():
    filename=filedialog.askopenfilename()
    pathlabel.config(text=os.path.basename(filename))
    print(filename)
    #aud.set(os.path.basename(filename))
    if(messagebox.askokcancel("Submit",filename)):
        aud.set(str(filename))
        file1=open ("location.txt",'w')
        file1.seek(0)
        file1.truncate(0)
        file1.write(str(filename))
        file1.close()
        
        
def display():
    disp= open("output.txt","r").read()
    out.set(str(disp))
    disp1= open("Final_emotion_detect.txt","r").read()
    final_out.set(str(disp1))
          

def media_analysis():
    
     VIDEO_FILE = aud.get()
     video = moviepy.editor.VideoFileClip(VIDEO_FILE)
     audio = video.audio
# Replace the parameter with the location along with filename
     audio.write_audiofile("media_audio.mp3")
     
     AUDIO_FILE = "media_audio.mp3"
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
    
     canvas= Canvas(tab4)
     #canvas.pack()
     img=PhotoImage(file="Wave.png")
     canvas.create_image(0,0,anchor=NW,image=img)
     canvas.image=img
     canvas.place(x=200,y=150)

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
    
     canvas1= Canvas(tab4)
     #canvas1.pack()
     img1=PhotoImage(file="spec.png")
     canvas1.create_image(0,0,anchor=NW,image=img1)
     canvas1.image=img1
     canvas1.place(x=200,y=300)
     
     #VIDEO_FILE="happy.mp4"
     #exec(open('C:/Users/Vishnu/Desktop/Final Project/Code/video/test.py').read())
     import subprocess
     subprocess.call(['ffmpeg', '-i', 'media_audio.mp3','audio.wav'])
     
     import speech_recognition as sr
     
     r = sr.Recognizer() 
  
     with sr.AudioFile('audio.wav') as source: 
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
     text_out.set(z['emotion']) 
     aud_out.set('male_happy')
    
    
#notebook creation
window.title("Emotion Analyzer")
tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab_control.add(tab1, text='Video')
tab_control.pack(expand=1, fill='both')
tab_control.add(tab4, text='Result')
tab_control.pack(expand=1, fill='both')

#video tab

V_label0=Label(tab1,text="Access Live WebCam: ",width=20,font=("arial",10,"bold"))
V_label0.place(x=20,y=25)

V_livecam=Button(tab1, text='LiveCam', width=10, command=live_cam.main)#command = some function
V_livecam.place(x=200,y=25)

V_label1=Label(tab1,text="Upload video file: ",width=20,font=("arial",10,"bold"))
V_label1.place(x=20,y=60)

file_loc= Label(tab1,textvariable=aud,font=("arial",10,"bold"),bg="white",height=1,width=40)
file_loc.place(x=200,y=60)

V_browse =Button(tab1, text='Browse', width=10, command=browsefunc) #command = some function
V_browse.place(x=200,y=90)

V_submit2=Button(tab1, text='Submit', width=10, command=media_file.main) #command = some function
V_submit2.place(x=300,y=90)

V_submit3=Button(tab1, text='Run Audio & text', width=15,command=media_analysis) #command = some function
V_submit3.place(x=400,y=90)

V_submit4=Button(tab1, text='Display O/P', width=10, command=display) #command = some function
V_submit4.place(x=300,y=130)

pathlabel= Label(window)
pathlabel.pack()

V_label2=Label(tab1,text="Emotion Extracted: ",width=20,font=("arial",10,"bold"))
V_label2.place(x=20,y=200)
#e1 = Entry(tab1)
#e1.place(x=200,y=53,width=250)

# out=open("output.txt","r").read()
V_label3=Label(tab1,textvariable=out,font=("arial",10,"bold"),bg="white",height=20,width=40,wraplength=250)
V_label3.place(x=200,y=200)

V_label4=Label(tab1,text="Final Emotion Detected: ",width=20,font=("arial",10,"bold"))
V_label4.place(x=20,y=550)


V_label5= Label(tab1,textvariable=final_out,font=("arial",10,"bold"),bg="white",height=2,width=40)
V_label5.place(x=200,y=550) 


#Result tab

videoFinalResult=Label(tab4,text="Video Emotion Detected: ",width=20,font=("arial",10,"bold"))
videoFinalResult.place(x=20,y=50)
videoFinalResultOutput=Label(tab4,textvariable=final_out,font=("arial",10,"bold"),bg="white",height=2,width=40,wraplength=250)
videoFinalResultOutput.place(x=200,y=53)

A_label1=Label(tab4,text="Audio Features  ",width=20,font=("arial",10,"bold"))
A_label1.place(x=20,y=100)

A_label2=Label(tab4,text="Waveform : ",width=20,font=("arial",10,"bold"))
A_label2.place(x=20,y=150)

A_label3=Label(tab4,text="Spectrogram : ",width=20,font=("arial",10,"bold"))
A_label3.place(x=20,y=300)

A_label4=Label(tab4,text="Audio Emotion Detected: ",width=20,font=("arial",10,"bold"))
A_label4.place(x=20,y=580)

A_label5= Label(tab4,textvariable=aud_out,font=("arial",10,"bold"),bg="white",height=2,width=40)
A_label5.place(x=200,y=580)

A_label6=Label(tab4,text="Text Emotion Detected: ",width=20,font=("arial",10,"bold"))
A_label6.place(x=20,y=620)

A_label6= Label(tab4,textvariable=text_out,font=("arial",10,"bold"),bg="white",height=2,width=40)
A_label6.place(x=200,y=620)



window.mainloop()
