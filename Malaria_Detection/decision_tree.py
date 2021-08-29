from .helper_functions import *
import numpy as np

#Decision Tree Functions
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
        # below static variables are used for better tree representation  
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
            question = "{} <= {}".format(feature_name, split_value)     # feature_name <= split_value
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree  | structure of  sub_tree = {question: [yes_anwer, no_answer]}
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace)
        
        # If the answers are the same, then there is no point in asking the question.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).

        # For example a case when max_depth is reached to during classification class with 
        # more no. of occurences is selected (and if something like this happens for both yes and no cases)
        # and that selected class comes as same for both cases
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)       #left node
            sub_tree[question].append(no_answer)        #right node
        
        return sub_tree


# 3. Make predictions
# 3.1 One example
def predict_example(example, tree):
    question = list(tree.keys())[0]                 # ['area_0', '<=' , '10.5']
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if float(example[feature_name]) <= float(value):
            answer = tree[question][0]              #store yes part
        else:
            answer = tree[question][1]              #store no part
    
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
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)      # tuple with single element
    return predictions