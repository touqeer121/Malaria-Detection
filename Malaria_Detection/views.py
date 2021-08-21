from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
import random
from pprint import pprint   
import cv2, os
from . import settings

#Helper Functions
# 1. Train-Test-Split
def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()        # stores indices: [0, 1, ...len(df)-1]
    test_indices = random.sample(population=indices, k=test_size)       #We get test_size random numbers from indices

    test_df = df.loc[test_indices]      # store those random indices into testing dataset
    train_df = df.drop(test_indices)    # and rest in training dataset
    
    return train_df, test_df


# 2. Distinguish categorical and continuous features
def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "Label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

# 3. Accuracy
def calculate_accuracy(predictions, labels):
    predictions_correct = predictions == labels
    accuracy = predictions_correct.mean()
    
    return accuracy


#Decision Tree Functions
# 1. Decision Tree helper functions 

# 1.1 Data pure?
def check_purity(data): 
    #data -> 2d numpy array

    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:                #all examples belong to same class
        return True
    else:
        return False

    
# 1.2 Classify
def classify_data(data):
    
    label_column = data[:, -1]                  #All rows of last column
     
    #For label_column = [1, 2, 1, 3, 3, 1] => unique_classes = [1, 2, 3] and  counts_unique_classes = [3, 1, 2]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)         

    index = counts_unique_classes.argmax()      #index of class having maximum frequency
    classification = unique_classes[index]      #classifying current data into class under which most examples are falling
    
    return classification


# 1.3 Potential splits?
def get_potential_splits(data, random_subspace):
    
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))    # excluding the last column which is the label
    
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    
    for column_index in column_indices:          
        values = data[:, column_index]              # All rows of current column
        unique_values = np.unique(values)
    
        potential_splits[column_index] = unique_values
    
    return potential_splits


# 1.4 Lowest Overall Entropy?
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()                   # p(infected) = no(infected) / (no(infected)+no(uninfected))
    entropy = sum(probabilities * -np.log2(probabilities))  # entropy = summission { p(i)* ( -log2 p(i) )}
     
    return entropy


def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy


def determine_best_split(data, potential_splits):
    
    # Here we basically test all potential splits and return the best one possible

    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# 1.5 Split data
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]                     # All rows of split_column

    type_of_feature = FEATURE_TYPES[split_column]
    
    # this dataset features
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]       # All rows where split_column_values <= split_value
        data_above = data[split_column_values >  split_value]       # All other rows
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


# 2. Decision Tree Algorithm
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None):
    
    # data preparations, since this is the first call to algorithm
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        #df-> dataframe,   data-> numpy 2d array
        data = df.values                    
    else:
        data = df           
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        #this dataset
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree  , structure of  sub_tree = {question: [yes_anwer, no_answer]}
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)       #left node
            sub_tree[question].append(no_answer)        #right node
        
        return sub_tree


# 3. Make predictions
# 3.1 One example
def predict_example(example, tree):
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

    
# 3.2 All examples of the test data
def decision_tree_predictions(test_df, tree):
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions


#Random Forest Classifier Functions
def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    return forest

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    return random_forest_predictions



def result(request):
    csv_folder = 'static/csv/'
    df = pd.read_csv(csv_folder+"/"+"dataset.csv")
    df = df[['area_0', 'area_1', 'area_2', 'area_3', 'area_4', 'Label']]
    
    # print(df)
    column_names = []
    for column in df.columns:
        name = column.replace(" ", "_")
        column_names.append(name)
    df.columns = column_names

    print(df.head())

    # activate below line to get same 'pseudo' random numbers each time
    # random.seed(0)
    train_df, test_df = train_test_split(df, test_size=0.2)

    forest = random_forest_algorithm(train_df, n_trees=5, n_bootstrap=800, n_features=5, dt_max_depth=5)
    print("FOREST START")
    for tree in forest:
        print(tree)
        print('\n-------------------------------------------------------------------------------------------')
    print("FOREST END")
    
    predictions = random_forest_predictions(test_df, forest)
    accuracy = calculate_accuracy(predictions, test_df.Label)

    print("My Model's Accuracy = {}".format(accuracy))

    response = {}
    return render(request, 'result.html', response)





#----------------------------------------------WEBPAGES------------------------------------------------#
def home(request):
    return render(request, 'home.html')

def preprocessData(request):
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

def result2(request):
    response = {}
    if request.method == "POST":        

        contours =  preprocessData(request)
        
        im_data = []
        for i in range(5):
            try:
                area = cv2.contourArea(contours[i])
                im_data.append(str(area))
            except:
                im_data.append("0")
        
        # cls = joblib.load("md_model")
        a= np.array([im_data])
        # ans = predict_example(a)
        # ans = cls.predict(a)

        response = {
            'result' : ans[0],
        }
        print("RESULT: ", response['result'])
    return render(request, 'result.html', response)