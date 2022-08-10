import sys
from pynput import keyboard
import time
import timeit
import torch



def on_press(key):
    #print(key, "is pressed")
    if (key == keyboard.Key.esc):
        print("aborting")
        return False
    elif key == keyboard.KeyCode.from_char("q"):
        print(" 'q' was pressed. aborting")
        return False
def on_release(key):
    #print(key, "is released")
    #print(key, "is released")
    #print(type(key), key, str(key))
    if (key == keyboard.Key.esc) or (str(key) ==  "q"):
        print("aborting")
        return False




listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
        )
listener.start()

while True:
    print("I am sleeping")
    time.sleep(1)
    if listener.is_alive():
        continue
    else:
        break
    

