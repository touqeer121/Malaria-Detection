import random

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
    predictions_correct = predictions == labels         #[True, True, False, ... , False]
    accuracy = predictions_correct.mean()
    
    return accuracy