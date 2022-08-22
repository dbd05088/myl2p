from PIL import Image
import os

base_path = './cifar100/train'
classes = os.listdir(base_path)
print(classes)

for klass in classes:
    path = base_path+ '/' + klass
    images = os.listdir(path)
    for img in images:
        img_path = path + '/' + img
        image = Image.open(img_path)
        print(image.size)
        
'''
image = Image.open("path/.../image.png")
image = image.resize((500,500),Image.ANTIALIAS)
image.save(fp="newimage.png")
'''