from PIL import Image
import os
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.competition_download_file('dogs-vs-cats-redux-kernels-edition','test.zip')
api.competition_download_file('dogs-vs-cats-redux-kernels-edition','train.zip')

from zipfile import ZipFile 
  
with ZipFile('test.zip', 'r') as zip: 
  

    zip.extractall() 

with ZipFile('train.zip', 'r') as zip:   

    zip.extractall()

def image_gs_scale(typ,num,scale):
    directory=os.getcwd()+'/train/'
    for i in range(num):
        img=Image.open(directory+typ+'.'+str(i)+'.jpg').convert('L')
        (wid,hei)=img.size
        wid*=scale
        hei*=scale
        img=img.resize((int(wid),int(hei)))
        img.save('gs_'+typ+str(i)+'.jpg')
        
        
def image_gs_size(typ,num,wid,hei):
    directory=os.getcwd()+'/train/'
    for i in range(num):
        img=Image.open(directory+typ+'.'+str(i)+'.jpg').convert('L').resize((wid,hei))
        img.save('gs_'+typ+str(i)+'.jpg')

image_gs_scale('cat',3,0.3)
image_gs_scale('dog',3,0.3)

















