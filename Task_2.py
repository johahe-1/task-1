
###########################################################################
############### IMPORTS AND DATA #########################################
###########################################################################

# Imports
import warnings # removes annoying error message
warnings.simplefilter(action='ignore', category=FutureWarning) # removes annoying error message
import pandas as pd # Matte
import numpy as np # Matte
import matplotlib as plt # Visualisering
import matplotlib.pyplot as plt # Visualisering

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
# PREPARING DATA FOR ALGORITHMS ###########################################
###########################################################################

# here the data is prepared for testing and training
 # hides the 'word'- and 'index'-columns
train_blind = train_head.drop(['word', 'index'], axis=1)
test_blind = test_head.drop(['word', 'index'], axis=1)


 # makes everything into the same datatype
test_blind = pd.DataFrame(data=test_blind, dtype=np.float64)
train_blind = pd.DataFrame(data=train_blind, dtype=np.float64)


# the algorithm below that removes NaN elements can segment the data, and can be used for other algorithms if needed

###########################################################################
# REMOVING NAN ELEMENTS ####### ###########################################
###########################################################################
# this process will be automated further for the whole dataframe, but this is a demo that shows how and that it works for single words

# find rows with NaN values anywhere in the DataFrame
rows_with_nan = train_head[train_head.isna().any(axis=1)]

# display the words of 'word' column for rows with NaN values
words_with_nan = rows_with_nan[['word','index']]
#print(words_with_nan)

# then this to be able to select and save an index
pick = 3 # <-- change to a number 0:6 to choose a word from words_with_nan

# what word was picked:
nan_word = words_with_nan['index'].iloc[pick]
print(words_with_nan.iloc[pick])

# here all instances of the word are stored in gest_all
gest_all = train_head[train_head['index'] == nan_word]
#print(gest_all)

# check if there are any NaN values in the DataFrame for frame 1 of 'word'
if gest_all[0:60].isna().any().any():
    print("DataFrame contains NaN elements")
else:
    print("DataFrame does not contain NaN elements")

# now to choose a frame: gest_f1, here the NaN-values of that frame for 'word' are located and replaced in train_head

# creates dictionaries to hold values for each index
x_values = {}
y_values = {}
z_values = {}

# loops through each instance
for i in range(len(gest_all)):
  gest = gest_all.iloc[i]
  gest_f1 = gest.iloc[0:60] # <-- this is the frame we choose

  # extracts index from column names
  for column in gest_f1.index:
    index = int(column.split('_')[1]) # chooses all column-indexes in interval

    # separates values by index and stores them in the corresponding dictionary
    if column.startswith('x'):
      x_values.setdefault(index, []).append(gest_f1[column])
    elif column.startswith('y'):
      y_values.setdefault(index, []).append(gest_f1[column])
    elif column.startswith('z'):
      z_values.setdefault(index, []).append(gest_f1[column])

'''
# print values for each index
for index in x_values.keys():
  print(f"x_{index}: {x_values[index]}")
  print(f"y_{index}: {y_values[index]}")
  print(f"z_{index}: {z_values[index]}")
  print()
'''
# now finally replace all NaN in the main dataframe
# iterate over each dictionary
for prefix, dictionary in [('x', x_values), ('y', y_values), ('z', z_values)]:
    # iterate over each column index
    for index, values in dictionary.items():
        # get the column name
        column_name = f'{prefix}_{index}'
        # replace NaN values with the mean of non-NaN values for the corresponding column in main dataframe
        train_head[column_name].fillna(np.nanmean(values), inplace=True)
        # for element in main dataframe all NaN are now replaced with the mean of respective list

# checks again if the word from the modified dataframe train_head contains any NaN
gest_all_filled = train_head[train_head['index'] == nan_word]
if gest_all_filled.isna().any().any():
    print("DataFrame still contains NaN elements")
else:
    print("DataFrame does not contain NaN elements anymore")

###########################################################################
############################## data prepp #################################
###########################################################################

# OBS train_head variabeln är alltså fylld nu (inga NaN)
    # VI BEHÖVER GÖRA DEN HÄR KODEN MER GENERELL
##LOOP!!

#skulle kunna förenklas så att den bara läser de två sista kolumnerna? inte prio

    #vi behöver ingen picker!! denna del behöver göras till en loop!!!
# find an instance of every word
instances = test_head[test_head.any(axis=1)]
word_tag = instances[['word','index']]
#print(instances)

# then this to be able to select and save an index
pick = 5 # <-- change to a number 0:27 to choose word
# what word was picked:
word = instances['index'].iloc[pick]
print(word_tag.iloc[pick])

# here all instances of the word are stored in gest_all
gest_all = test_head[test_head['index'] == word]
#print(gest_all)
##LOOP!!

# KODEN NEDAN KAN FÖRBLI DENSAMMA, VI KANSKE TILL OCH MED KAN SKICKA UPP ORDEN ISTÄLLET?
#creates dictionaries to hold values for each index
x_values = {}
y_values = {}
z_values = {}

# loops through each instance
for i in range(len(gest_all)):
  gest = gest_all.iloc[i]
  gest_f1 = gest.iloc[0:60] # <-- this is the frame we choose

  # extracts index from column names
  for column in gest_f1.index:
    index = int(column.split('_')[1]) # chooses all column-indexes in interval

    #istället för dictionaries gör vi dem till arrays direkt? för knn iallafall
    #borde bli simplare att stega igenom

    # separates values by index and stores them in the corresponding dictionary
    if column.startswith('x'):
      x_values.setdefault(index, []).append(gest_f1[column])
    elif column.startswith('y'):
      y_values.setdefault(index, []).append(gest_f1[column])
    elif column.startswith('z'):
      z_values.setdefault(index, []).append(gest_f1[column])


# print values for each index
for index in x_values.keys():
  print(f"x_{index}: {x_values[index]}")
  print(f"y_{index}: {y_values[index]}")
  print(f"z_{index}: {z_values[index]}")
  print()

#
'''
#################### knn #######################################
#TA REDA PÅ k
#har för mig att det finns en regel att knn vill ha sqrt(datapoints) som k
    #antingen om vi utgår från en algoritm som hittar minsta 'word'-arrayen och tar roten..
    #..ur den, det är illafall en början
    #OM det inte fungerar, så kan vi ta sqrt(word_array)=k för varje ord, men blir...
    #... betydligt mindre optimerat, vilket knn redan är för stora mängder data
#############################################################

# + alla andra classifiers

# vi delar in orden i arrays i test_blind också
# hmm använda datan för vinklar också här...?
# FÖR VARJE " i in 0,'index_max' -> [insert classifier] "
# vi gör sedan om visualiseringen så att den visar ordet classifiern gissat och faktiska ordet
# ^ kommer vara användbart för Task 3 också, när vi ska utvärderar vilken  som är bäst

###########################################################################
############################## VISUALISERING ##############################
###########################################################################

# changes typ from dataframe to array
gest_all = gest_all_filled.to_numpy() #<-- gest_all for gestures with NaN, gest_all_filled for replaced NaN

Kropp = [11, 10, 1, 0]

H_arm = [1, 3, 5, 7, 9]
V_arm = [1, 2, 4, 6, 8]

H_ben = [11, 12, 14, 16, 18]
V_ben = [11, 13, 15, 17, 19]

# empty figure to plot in
fig = plt.figure(figsize=(12, 7))

for j in range(0,len(gest_all)): # plots all instances of gesture
    gest = gest_all[j]
    gest_f1 = gest[0:60]  #<-- choose frame 1

    x = gest_f1[0::3]
    y = gest_f1[1::3]
    z = gest_f1[2::3]
    #print(x,y,z, sep='\n') #prints coordinates

    ax = fig.add_subplot(5, 5, j+1, projection='3d')
    ax.scatter(x, y, z, s = 2)
    ax.plot(x[Kropp], y[Kropp], z[Kropp], label = "Kropp")
    ax.plot(x[H_arm], y[H_arm], z[H_arm], label = "Höger Arm")
    ax.plot(x[V_arm], y[V_arm], z[V_arm], label = "Vänster Arm")
    ax.plot(x[H_ben], y[H_ben], z[H_ben], label = "Höger Ben")
    ax.plot(x[V_ben], y[V_ben], z[V_ben], label = "Vänster Ben")

    ax.view_init(elev=-70, azim=90)
    ax.legend(fontsize=2)

plt.tight_layout()
plt.show()
'''
