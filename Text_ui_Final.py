from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import os
import paralleldots
import json
from tkinter import messagebox
import speech_recognition as sr
window=Tk()

window.geometry("600x600")
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
    
def aud_click():
    
    AUDIO_FILE = aud.get()
  
    # use the audio file as the audio source 
  
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


    api_key   = "ANxhDZ8aioVsDqhRDMkqgNgkNRpzjxVdqGFPhBPtYrQ"
    paralleldots.set_api_key( api_key )
    text=r.recognize_google(audio)
    print(text)
    
    print( "\nEmotion" )
    print( paralleldots.emotion(text))
    print( "\nAbuse" )
    print( paralleldots.abuse(text))
    print( "\nSentiment" )
    print( paralleldots.sentiment(text))
    
    y=json.dumps(paralleldots.emotion(text))
    z=json.loads(y)
    out.set(str((paralleldots.emotion(text)))+"\n"+str((paralleldots.abuse(text)))+"\n"+str((paralleldots.sentiment(text))))
    print(z['emotion'])
    final_out.set(z['emotion'])

def txt_click():
    api_key   = "ANxhDZ8aioVsDqhRDMkqgNgkNRpzjxVdqGFPhBPtYrQ"
    paralleldots.set_api_key( api_key )

    text=textentry.get("1.0","end-1c")
    print( "\nEmotion" )
    print( paralleldots.emotion(text))
    print( "\nAbuse" )
    print( paralleldots.abuse(text))
    print( "\nSentiment" )
    print( paralleldots.sentiment(text))
    y=json.dumps(paralleldots.emotion(text))
    z=json.loads(y)
    out.set(str((paralleldots.emotion(text)))+"\n"+str((paralleldots.abuse(text)))+"\n"+str((paralleldots.sentiment(text))))
    print(z['emotion'])
    final_out.set(z['emotion'])

    
#notebook creation
window.title("Emotion Analyzer")
tab_control = ttk.Notebook(window)
tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text='Text')
tab_control.pack(expand=1, fill='both')



#text tab
    
T_label0=Label(tab3,text="Upload Audio File: ",width=20,font=("arial",10,"bold"))
T_label0.place(x=20,y=25)

file_loc= Label(tab3,textvariable=aud,font=("arial",10,"bold"),bg="white",height=1,width=40)
file_loc.place(x=200,y=25)

T_browse=Button(tab3, text='Browse', width=10, command=browsefunc) #command = some function
T_browse.place(x=200,y=50)
T_submit =Button(tab3, text='Submit', width=10, command=aud_click) #command = some function
T_submit.place(x=300,y=50)

pathlabel= Label(window)
pathlabel.pack()

T_label1=Label(tab3,text="Enter Text: ",width=20,font=("arial",10,"bold"))
T_label1.place(x=20,y=80)
T_entry = Text(tab3,height=5, width=40)
T_entry.place(x=200,y=80)

T_submit2=Button(tab3, text='Submit', width=10, command=txt_click) #command = some function
T_submit2.place(x=200,y=170)

T_label2=Label(tab3,text="Text Emotion Extracted: ",width=20,font=("arial",10,"bold"))
T_label2.place(x=20,y=200)

T_label3=Label(tab3,textvariable=out,font=("arial",10,"bold"),bg="white",height=15,width=40,wraplength=250)
T_label3.place(x=200,y=200)

T_label4=Label(tab3,text="Final Emotion Detected: ",width=20,font=("arial",10,"bold"))
T_label4.place(x=20,y=470)

T_label5= Label(tab3,textvariable=final_out,font=("arial",10,"bold"),bg="white",height=2,width=40)
T_label5.place(x=200,y=470)

window.mainloop()
