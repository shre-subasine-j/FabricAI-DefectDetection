#programming_fever
import cv2
import numpy as np
import pandas as pd
import os
from time import sleep
xt=int(input("Enter 1 for Camera 2 for direct image")) 
if xt==1:
    camera = cv2.VideoCapture(0)
    sleep(3)
    return_value, image = camera.read()
    cv2.imwrite('main.png', image)
    sleep(3)
    del(camera)
    img_path = "main.png"
elif xt==2:
	sd=input("ENTER IMAGE NAME")
	img_path=sd
img = cv2.imread(img_path)
img=cv2.resize(img,(700,500))
text=""
clicked = False
r = g = b = xpos = ypos = 0

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

#function to get x,y coordinates of mouse double click
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
cv2.namedWindow('color detection by programming_fever')
cv2.setMouseCallback('color detection by programming_fever',draw_function)

while(1):

    cv2.imshow("color detection by programming_fever",img)
    if (clicked):
   
        #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        #Creating text string to display( Color name and RGB values )
        text = getColorName(r,g,b)
        
        #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)
        print(text)
        break

        #For very light colours we will display text in black colour
        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            print(text)
        clicked=False

    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()

import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
#%matplotlib inline
# ----- LOAD SAVED MODEL -----
#json_file = open('model.json', 'r')     
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk.")
import tensorflow as tf
loaded_model = tf.keras.models.load_model("CNN.model")




CATEGORIES = ["animal","check","chevron","diamond","floral","heart","leaf","paisley","polka dots","stripes"]
import numpy as np
from tensorflow.keras.preprocessing import image
#test_image = image.load_img(img_path, target_size = (50, 50))
test_image = cv2.imread(img_path, 0)
test_image=cv2.equalizeHist(test_image)
test_image = cv2.resize(test_image, (50, 50))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image=test_image.reshape(-1, 50, 50, 1)
result = loaded_model.predict(test_image)

prediction = list(result[0])
print(prediction)
print(CATEGORIES[prediction.index(max(prediction))])

s="The colour is "+str(text)+" The shape is "+str(CATEGORIES[prediction.index(max(prediction))])
file = open("main.txt", "w") 
file.write(s) 
file.close() 
with open('main.txt') as f:
    lines = f.readlines()
import win32com.client
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Speak(lines)
