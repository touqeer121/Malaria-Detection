from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
from pprint import pprint   
import cv2, os
from . import settings
from .helper_functions import *
from .decision_tree import *
from .random_forest_classifier import *

def processImage(request):
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

    return contours
    
#----------------------------------------------WEBPAGES------------------------------------------------#
def home(request):
    return render(request, 'home.html')

def result(request):
    response = {}
    if request.method == "POST":        

        contours =  processImage(request)        
        im_data = []
        for i in range(5):
            try:
                area = cv2.contourArea(contours[i])
                im_data.append(str(area))
            except:
                im_data.append("0")
        
        # cls = joblib.load("md_model")
        data= np.array([im_data])
        # ans = cls.predict(data)

        csv_folder = 'static/csv/'
        df = pd.read_csv(csv_folder+"/"+"dataset.csv")
        df = df[['area_0', 'area_1', 'area_2', 'area_3', 'area_4', 'Label']]
    
        column_names = []
        for column in df.columns:
            name = column.replace(" ", "_")
            column_names.append(name)
        df.columns = column_names

        # activate below line to get same 'pseudo' random numbers each time
        # random.seed(0)
        train_df, test_df = train_test_split(df, test_size=0.2)

        forest = random_forest_algorithm(train_df, n_trees=5, n_bootstrap=800, n_features=5, dt_max_depth=5)

        # for tree in forest:
        #     pprint(tree)
        
        new_df = pd.DataFrame({'area_0': data[:, 0], 'area_1': data[:, 1], 'area_2': data[:, 2], 'area_3': data[:, 3], 'area_4': data[:, 4]})
        # print("DATAFRAME:\n")
        # print(new_df)
        predictions = random_forest_predictions(new_df, forest)
        all_predictions = random_forest_predictions(df, forest)
        accuracy = calculate_accuracy(all_predictions, df.Label)
        print("My Model's Accuracy = {}".format(accuracy))

        response['result'] = predictions[0]
        
        print("RESULT: ", response['result'])
    return render(request, 'result.html', response)