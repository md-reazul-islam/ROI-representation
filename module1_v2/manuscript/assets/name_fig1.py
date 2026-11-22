"""Rename the labels of the user's Figure 1 (blurred) to the paper's names; writes Figure1_blurred_named.png."""
from PIL import Image, ImageDraw, ImageFont
import matplotlib, os, numpy as np
fd=os.path.join(matplotlib.get_data_path(),'fonts/ttf')
BOLD=os.path.join(fd,'DejaVuSans-Bold.ttf'); REG=os.path.join(fd,'DejaVuSans.ttf')
im=Image.open('Figure1_blurred.png').convert('RGB'); a=np.asarray(im); d=ImageDraw.Draw(im)
def put(box,bgpt,new,fontfile,size,color,center=False):
    x0,y0,x1,y1=box
    d.rectangle(box,fill=tuple(int(v) for v in a[bgpt[1],bgpt[0]]))
    f=ImageFont.truetype(fontfile,size); w=d.textlength(new,font=f); bb=f.getbbox(new)
    d.text(((x0+x1-w)/2 if center else x0+2,(y0+y1)/2-(bb[1]+bb[3])/2),new,font=f,fill=color)
for x0,x1 in ((296,334),(765,803),(1185,1223)):
    put((x0,371,x1,391),(x0+1,372),'new',BOLD,14,(255,255,255),center=True)
put((474,429,713,449),(474,429),'PF-ROI: 9.86 switches/s',BOLD,15,(235,80,20))
put((474,513,676,532),(474,513),'NeoSTAR (D3): 0.24 switches/s',BOLD,15,(30,100,220))
put((1549,577,1690,598),(1549,577),'Learning-free',REG,16,(25,25,50))
im.save('Figure1_blurred_named.png')
