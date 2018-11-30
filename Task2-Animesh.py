# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:07:45 2018
@author: Animesh
"""
import cv2
import numpy as np
import os
n = 7 #Order of Gaussian Kernel
pad_factor = int((n-1)/2) #Number of Padding

sigma=[[1/np.sqrt(2),1,np.sqrt(2),2,2*np.sqrt(2)],[np.sqrt(2),2,2*np.sqrt(2),4,4*np.sqrt(2)],[2*np.sqrt(2),4,4*np.sqrt(2),8,8*np.sqrt(2)],[4*np.sqrt(2),8,8*np.sqrt(2),16,16*np.sqrt(2)]]

def imgread():
  image = cv2.imread("task2.jpg", 0) #Read Image as Numpy Array
  a = []
  for i in range(0,len(image)):
      a.append(image[i])
  return a

def imgdisplay(img):
    img = np.array(img)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Create Image Display Window
    cv2.imshow('image', img) # Display Image
    cv2.waitKey(0)
    cv2.destroyAllWindows() #Destroy Window

def pad(m,n,img):
    temp = [[0 for x in range(n)] for y in range(m)]
    for i in range(0,len(img)):
        for j in range(0,len(img[i])): #Zero Padded Array with m+2 rows and n+2 column
            temp[i+pad_factor][j+pad_factor] = img[i][j] # Temp holds the complete padded array list. We shall use this for convolution. 
    return temp

def gauss_func(x,y,sigma):
    f = (np.exp(-((x**2+y**2)/2*(sigma**2))))/(2*np.pi*(sigma**2))
    return f

def gauss_kernel(n, sigma): 
    kernel = [[0 for x in range(n)] for y in range(n)]
    y = 3
    for i in range(0,n):
        x = -3
        for j in range(0,n):
            kernel[i][j] = gauss_func(x,y,sigma)
            x = x+1
        y = y-1
    return kernel

def normalize(kernel):
    sum = 0
    for i in range(0,len(kernel)):
        for j in range(0,len(kernel[i])):
            sum = sum + kernel[i][j]
    
    for i in range(0,len(kernel)):
        for j in range(0,len(kernel[i])):
            kernel[i][j] = kernel[i][j]/sum
    return kernel

def convolve(m,n,temp,kernel):
    conv = [[0 for x in range(n)] for y in range(m)]
    for i in range(0,len(conv)):
        for j in range(0,len(conv[i])): #Inner Product Computations
            sum=0
            for k in range(0,len(kernel)):
                for l in range(0,len(kernel[k])):
                    sum = sum + kernel[k][l]*temp[i+k][j+l]
                conv[i][j] = sum
    return conv

def resize_image(img):
    resized = []
    for i in range(0,len(img)): 
        t = []
        for j in range(0,len(img[i])):
            if (i%2==0) and (j%2==0):
                t.append(img[i][j]) #t is a 1D array
        if i%2==0:
            resized.append(t)
    return resized

def compute_intensities(img,a,b,temp,h,w):
    intensities = [] # 3D List stores each image after convolution
    for i in range(0,a): #For each Octave
        for j in range(0,b): #For each value of sigma
            sample_kernel = gauss_kernel(n,sigma[i][j]) #Kernel for a particular value of sigma
            sample_kernel = normalize(sample_kernel) #Normalize Kernel
            #print(np.asarray(sample_kernel))
            conv = convolve(h,w,temp,sample_kernel) #Intensity Matrix after Convolution
            intensities.append(conv)
        img = resize_image(img) #Returns Resized Image
        h = len(img)
        w = len(img[0])
        temp = pad(h + 2*pad_factor, w + 2*pad_factor, img) #Padded Image Matrix
    return intensities

def compute_dog(img):
    dog = []
    for i in range(0,len(img)-1): #Number of Images
        d2 = []
        for j in range(0,len(img[i])): #Number or Rows in Image
            d1 = []
            for k in range(0,len(img[i][j])): #Number or Columns in Image
                #lenth=len((img[i][j]))
                diff = img[i][j][k] - img[i+1][j][k]
                d1.append(diff) # 1D Array
            d2.append(d1) # 2D Array 
        dog.append(d2) #3D Array of 1 row
    return dog

def handle_resize_exception(img):
    nom = 0 # Number of intensity matrices
    dog = []
    while nom < len(img):
        row = [img[i] for i in range(nom,nom + 5)] #Rowwise 3D Matrix of Images
        rowwise_dog = compute_dog(row) #Sends each Image Set rowwise to avoid order mismatch due to resizing
        nom = nom + 5
        dog.append(rowwise_dog)
    return dog

def find_keypoints(dog1,dog2,dog3,octave):
    dict = {'xpixel': [], 'ypixel': [], 'octave': []}
    for i in range(0,len(dog1)-2):
        for j in range(0,len(dog1[i])-2):
            temp = []
            for x in range(0,3):
                for y in range(0,3):
                    temp.append(dog1[i+x][j+y])
                    temp.append(dog2[i+x][j+y])
                    temp.append(dog3[i+x][j+y])
            central = dog2[i+1][j+1]
            high = max(temp)
            low = min(temp)
            if central == high or central == low: 
                #Then central is a keypoint. Need to preserve i,j,octave values
                dict['xpixel'].append(i+1)
                dict['ypixel'].append(j+1)
                dict['octave'].append(octave)
    return dict #Returns Dictionary of all Keypoints

def collect_keypoints(dog):
    xpoints = []
    ypoints = []
    octs = []
    for i in range(0,len(dog)):
        for j in range(0,2):
            keypoints = find_keypoints(dog[i][j],dog[i][j+1],dog[i][j+2],i+1) #Returns Dictionary of all keypoints
            xpoints.append(keypoints["xpixel"])
            ypoints.append(keypoints["ypixel"])
            octs.append(keypoints["octave"])
    return xpoints, ypoints, octs
    

def maptoimage(img,x,y,o):
#    x = key["xpixel"] #Returns list of all x coordinate keypoints
#    y = key["ypixel"] #Returns list of all y coordinate keypoints
#    o = key["octave"] #Returns list of all Octaves
    for i in range(0,len(o)):
        for j in range(0,len(o[i])):
            if o[i][j] == 1:
                img[x[i][j]][y[i][j]] = 255
            elif o[i][j] == 2:
                img[2*(x[i][j])][2*(y[i][j])] = 255
            elif o[i][j] == 3:
                img[4*(x[i][j])][4*(y[i][j])] = 255
            else:
                img[8*(x[i][j])][8*(y[i][j])] = 255
    return img

def write_images(intensities):
    j = 0
    for i in range(0,len(intensities)): 
        npimage=np.array(intensities[i])
        if i>=5 and i%5==0:
            j = j+1
        name = "octave_{0}image_{1}".format(j+1,i+1)
        cv2.imwrite('./images/'+ name +'.png',npimage)
        
def write_dogs(dog):
    for i in range(0,len(dog)): 
        for j in range(0,len(dog[i])): 
            npimage=np.array(dog[i][j])
            name = "octave_{0}dog_{1}".format(i+1,j+1)
            cv2.imwrite('./dogs/'+ name +'.png',npimage)
    
#Main Starts here --------------------------

img = imgread()

#print(np.array(img))
#print()

h = len(img)
w = len(img[0])

temp = pad(h + 2*pad_factor, w + 2*pad_factor, img) #Padded Image Matrix
#print (np.array(temp))


#print(np.array(conv))
#imgdisplay(conv)

a = len(sigma)
b = len(sigma[0])

intensities = compute_intensities(img,a,b,temp,h,w)
dog = handle_resize_exception(intensities)
xpoints,ypoints,octs = collect_keypoints(dog)
img = maptoimage(img,xpoints,ypoints,octs)

dirs = ['images','dogs']
for d in dirs:
    if not os.path.exists(d):
        os.makedirs('./'+d+'/') #Creates folders in working directories

write_images(intensities)
write_dogs(dog)
img = np.array(img)
cv2.imwrite('final_image.jpg',img)
#imgdisplay(img)