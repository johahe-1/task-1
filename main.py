###########################################################################
############### IMPORTS AND DATA #########################################
###########################################################################

# Import packages
import warnings # removes annoying error message
warnings.simplefilter(action='ignore', category=FutureWarning) # removes annoying error message
warnings.filterwarnings("ignore") # ignorera RuntimeWarning error och annat från bootstrap
import pandas as pd # Matte
import numpy as np # Matte
import matplotlib.pyplot as plt # Visualisering
from scipy.stats import bootstrap

# Import filer
import dataplot
from RandomForest import random_forest_2
import boo as boo
import knn

# path to data from directory
train = pd.read_csv('train-final.csv',encoding='latin-1',sep=',')
test = pd.read_csv('test-final.csv',encoding='latin-1',sep=',')

###########################################################################
############### GER DATAFRAMSEN HEADER MED VAD KOLUMNERNA INNEHÅLLER#######
###########################################################################

# all xyz and angle coordinates
num_sets_space = 60
num_cols_per_set_space = 3

num_sets_angle = 60
num_cols_per_set_angle = 1

# generates the coord with appropriate index and creates a 1d dataframe
space_names = [f'{coord}_{i}' for i in range(1, num_sets_space + 1) for coord in ['x', 'y', 'z']]
angle_names = [f'{coord}_{i}' for i in range(1, num_sets_angle + 1) for coord in ['v']]
index_names = ['word', 'index']

# combine the lists into one row DataFrame
names = space_names + angle_names + index_names

# reshape dataframes into two-dimensional arrays
reshaped_test = np.reshape(test.values, (-1, test.shape[-1]))
reshaped_train = np.reshape(train.values, (-1, train.shape[-1]))

# create a DataFrame with names as index and reshaped_test as data
test_head = pd.DataFrame(reshaped_test, columns=names)
train_head = pd.DataFrame(reshaped_train, columns=names)



###########################################################################
# REMOVING NAN ELEMENTS ####### ###########################################
###########################################################################

# stores both dataframes in a dictionary so that they can be accessed in a loop
all_data = [train_head, test_head ]


def nanremove(data):
    for df in range(len(data)):
        '''
        #checks if dataframes has NaN-elements
        if all_data[df].isna().any().any():
            print("DataFrame contains NaN elements")
        else:
            print("DataFrame does not contain NaN elements")
        '''
        # find rows with NaN values anywhere in the DataFrame
        rows_with_nan = data[df][data[df].isna().any(axis=1)]

        # display the words of 'word' column for rows with NaN values
        words_with_nan = rows_with_nan[['word','index']]
        #print(words_with_nan)

        # then this to be able to select and save an index
        for idx, row in rows_with_nan.iterrows():
            # What word was picked:
            nan_word = row['index']  # This gets the 'word' from the current row
            # print(f"Word: {nan_word}, Index: {row['index']}")

        # here all instances of the word are stored in gest_all
            gest_all = data[df][data[df]['index'] == nan_word]
            # print(gest_all)

        # check if there are any NaN values in the DataFrame for frame 1 of 'word'
            #if gest_all[0:60].isna().any().any():
                #print("DataFrame contains NaN elements")
            #else:
                #print("DataFrame does not contain NaN elements")

        # creates dictionaries to hold values for each index
            x_values = {}
            y_values = {}
            z_values = {}

        # loops through each instance
            for i in range(len(gest_all)):
                gest = gest_all.iloc[i]
                gest_xyz = gest.iloc[0:120] # <-- this is the frame we choose

            # extracts index from column names
                for column in gest_xyz.index:
                    index = int(column.split('_')[1]) # chooses all column-indexes in interval

            # separates values by index and stores them in the corresponding dictionary
                    if column.startswith('x'):
                        x_values.setdefault(index, []).append(gest_xyz[column])
                    elif column.startswith('y'):
                        y_values.setdefault(index, []).append(gest_xyz[column])
                    elif column.startswith('z'):
                        z_values.setdefault(index, []).append(gest_xyz[column])

        # now finally replace all NaN in the main dataframe
        # iterate over each dictionary
        for prefix, dictionary in [('x', x_values), ('y', y_values), ('z', z_values)]:
            # iterate over each column index
            for index, values in dictionary.items():
                # get the column name
                column_name = f'{prefix}_{index}'
                # replace NaN values with the mean of non-NaN values for the corresponding column in main dataframe
                data[df][column_name].fillna(np.nanmean(values), inplace=True)
                # for element in main dataframe all NaN are now replaced with the mean of respective list
        '''
        # checks again if the modified dataframe train_head contains any NaN
        if all_data[df].isna().any().any():
            print("DataFrame still contains NaN elements")
        else:
            print("DataFrame does not contain NaN elements anymore")
        '''
    return data


# now we have labled and unlabeled data without any nan-values
all_data_processed = nanremove(all_data)

train_processed_word = all_data_processed[0]
test_processed_word = all_data_processed[1]
train_processed = all_data_processed[0].drop(columns=['word'])
test_processed = all_data_processed[1].drop(columns=['word'])


#bootstrap_samples = boo.samp_rep(train_processed, 10, 100)
#data, labels = boo.bootstrap_dict(train_processed, 10, 100)

#print(data,labels)

#knn.rknn(train_processed, test_processed, 50)

random_forest_2(train_processed,test_processed)

#not very diverse distribution
#bootstrap_samples = boo.samp_rep(train_processed, 5, 100)
#boo.boot_stats(bootstrap_samples)

# wider distribution but unable to extract samples
#boo.bootstrapper(train_processed, 5)
