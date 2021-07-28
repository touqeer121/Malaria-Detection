from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import cv2, os
from . import settings
def home(request):
    return render(request, 'home.html')

def result(request):
    response = {}
    if request.method == "POST":
        img = request.FILES['cell_image']
        img_extension = os.path.splitext(img.name)[1]

        images_folder = 'static/images/'
        if not os.path.exists(images_folder):
            os.mkdir(images_folder)
        
        img_save_path = images_folder+ '/input'+ img_extension
        print("ISP", img_save_path)
        with open(img_save_path, 'wb+') as f:
            for chunk in img.chunks():
                f.write(chunk)

        im = cv2.imread(img_save_path)
        im = cv2.GaussianBlur(im,(5,5),2)
        im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(im_gray,127,255,0)
        contours,_ = cv2.findContours(thresh,1,2)
        
        im_data = []
        for i in range(5):
            try:
                area = cv2.contourArea(contours[i])
                im_data.append(str(area))
            except:
                im_data.append("0")
        
        cls = joblib.load("md_model")
        a= np.array([im_data])
        ans = cls.predict(a)

        response = {
            'result' : ans[0],
        }
        print("RESULT: ", response['result'])
    return render(request, 'result.html', response)