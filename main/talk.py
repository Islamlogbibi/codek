import gtts as gt
import pygame as pg
import os
def Talk(text):
    tts = gt.gTTS(text=text, lang='en')
    tts.save("response.mp3")
    pg.mixer.init()
    pg.mixer.music.load('response.mp3')
    pg.mixer.music.play()
    while pg.mixer.music.get_busy():
        pg.time.Clock().tick(10)
    os.system("mpv --no-video response.mp3") 
    os.remove("response.mp3") 

