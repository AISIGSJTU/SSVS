import cv2
import numpy as np
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    CLAHE,
    RandomRotate90,
    Rotate,
    IAAPiecewiseAffine,
    IAAPerspective,
    RandomContrast,#limit=0.2
    RandomBrightness,#limit=0.2
    GaussNoise,#var_limit=50
    Normalize#Default:mean,std of ImageNet 2012 {mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]}
)
from tqdm import trange
PATH='datasets/SSV/trainA'
def rectangle(x,y,w,h,img,flag=-1,ite=3):
    epsx=np.random.randint(-25,25)
    epsy=np.random.randint(-25,25)
    if x<0:
        x=0
    if y<0:
        y=0
    if x+w>511:
        x2=511
    else:
        x2=x+w
    if y+h>511:
        y2=511
    else:
        y2=y+h
    img[int(x):int(x2),int(y):int(y2)]=255
    flag*=-1
    ite-=1
    if ite>0:
      if flag==1:
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*h/6))
        img=rectangle(x-w2,y+h/3,w2,h2,img,flag,ite)
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*h/6))
        img=rectangle(x,y+eps,w2,h2,img,flag,ite)
      else:
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*w/6))
        img=rectangle(x+eps,y-h2,w2,h2,img,flag,ite)
        w2=np.random.randint(int(h/4),int(3*h/4+1))
        h2=np.random.randint(int(w/4),int(3*w/4+1))
        eps=np.random.randint(1,int(5*w/6))
        img=rectangle(x+eps,y+h,w2,h2,img,flag,ite)
    return img



def nextx(x,y,w,h):
    x=np.random.randint(x,x+w)
    y=np.random.randint

for i in trange(0,20):
  img=np.zeros((512,512))
  x=np.random.randint(10,250)
  w=np.random.randint(15,20)#(15,25)
  y=np.random.randint(2,80)
  h=np.random.randint(350,450)
  ite=np.random.randint(3,5)
  img=rectangle(x,y,w,h,img,-1,ite)
  '''
  ipath = PATH +str(i).zfill(5)+'.png'
  cv2.imwrite(str(ipath),img)
  img[int(x-y/2):x,int(y+h/3):int(y+h/3+w/2)]=255
  img[x+w:int(x+w+h/2),int(y+h-h/4):int(y+h-h/4+w/2)]=255
  '''
  aug=Compose([IAAPiecewiseAffine(scale=(0.09, 0.13), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', always_apply=False, p=1),Rotate(limit=30, p=0.5)], p=1)
  img = aug(image=img)['image']
  ipath = PATH +str(i).zfill(5)+'.png'
  cv2.imwrite(str(ipath),img)
