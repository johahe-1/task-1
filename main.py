
# BASBLOCK
# Ladda från github


# moduler
# machine-learning

import pandas as pd

#matte
import numpy as np

#visualisering
#import matplotlib as plt
import matplotlib.pyplot as plt
# visualisering
# import matplotlib as plt
import matplotlib.pyplot as plt
# matte
import numpy as np
import pandas as pd

#datan
  #används endast för att TRÄNA modellen
train = pd.read_csv('https://raw.githubusercontent.com/johahe-1/task-1/main/train-final.csv',encoding='latin-1',sep=',')
  #användst endast för att TESTA modellen
test = pd.read_csv('https://raw.githubusercontent.com/johahe-1/task-1/main/test-final.csv',encoding='latin-1',sep=',')




### GER DATAFRAMSEN HEADER MED VAD KOLUMNERNA INNEHÅLLER###

# alla xyz- och vinkel-koord
num_sets_space = 60
num_cols_per_set_space = 3

num_sets_angle = 60
num_cols_per_set_angle = 1

# Generates the coord with appropriate index and creates a 1d dataframe
space_names = [f'{coord}_{i}' for i in range(1, num_sets_space + 1) for coord in ['x', 'y', 'z']]
angle_names = [f'{coord}_{i}' for i in range(1, num_sets_angle + 1) for coord in ['v']]
index_names = ['word', 'index']

# Combine the lists into one row DataFrame
names = space_names + angle_names + index_names

# Reshape dataframes into two-dimensional arrays
reshaped_test = np.reshape(test.values, (-1, test.shape[-1]))
reshaped_train = np.reshape(train.values, (-1, train.shape[-1]))

# Create a DataFrame with names as index and reshaped_test as data
test_head = pd.DataFrame(reshaped_test, columns=names)
train_head = pd.DataFrame(reshaped_train, columns=names)

test_head.head()

#print(train_head['index'].nunique)
for x in range(1,30):
  #hitta instanser av tecknade index i datan
  gest_all = train_head[train_head['index'].str.endswith(x)] #<-- i den första välj 'word' eller 'index', i den andra ordet eller indexen för ordet

  if gest_all.isnull().any().any():
    print(gest_all.isnull.any())





 #### PREPPAR DATAN FÖR CLASSIFYERS ####

# Tar bort ord-tags och skapar blind-versioner som ska dölja svaren för classifyers
#train_blind = train.drop(['bye', '5'], axis=1)
#test_blind = test.drop(['wind', '28'], axis=1)

# gör så att alla blir samma datatyp
#test_blind = pd.DataFrame(data=test_blind, dtype=np.float64)
#train_blind = pd.DataFrame(data=train_blind, dtype=np.float64)



for x in range(1,train_head['index'].nunique):
  #hitta instanser av tecknade index i datan
  gest_all = train_head[train_head['index'].str.endswith(x)] #<-- i den första välj 'word' eller 'index', i den andra ordet eller indexen för ordet

  if gest_all.isnull().any().any():
    print(gest_all.isnull.any())



# Display the resulting DataFrame
#print(gest_all)

#for i in range(0,len(gest_all)):
#  gest = gest_all.iloc[i]
#
#  gest_f1 = gest.iloc[0:60]
#
#  x = gest_f1.iloc[0::3]
#  y = gest_f1.iloc[1::3]
#  z = gest_f1.iloc[2::3]
#
#  y_3 = y.iloc[2::20]
#  mean = y_3.mean()
#  y_3 = y_3.fillna(mean)
#  print(y_3)
  # Replace NaN values with 0
  #y_3_nan_zero = y_3.fillna(0)

# Calculate the mean of all non-NaN values
  #mean_non_nan = y_3_nan_zero.values.mean()
    #for j in gest if gest[j]





# Print the resulting DataFrame
#print(gest)

#nan_rows = x.isna().any(axis=1)
#print(nan_rows)


####### KNN ######

# Instantiate the KNN classifier with k=3
#knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
#knn.fit(X_train, y_train)

# Predict the response for test dataset
#y_pred = knn.predict(X_test)

# Model accuracy
#accuracy = knn.score(X_test, y_test)
#print('KNN model accuracy: ', accuracy)
#type(iris)


#### VISUALISERING ####

#försöker skapa något som tar ut alla iterationer av en gest.
#Så nu behövs inte min sorterade fil :(

#gör om dataframen till en array
data_array = train.to_numpy()

#print(data_array)

#dear upp datan inför visualisering
def chooser(x):
  choice = 1 #Här väljer man vilken gest man vill ha
  return x[-1] == choice #kollar sista värdet i listan för att se om det är rätt gest

gest_all = list(filter(chooser,data_array)) #filter(funktion, iterable) tar en funktion och en lista med data och kör den genom en funktion och listar de objekt som får ett True
#print(gest_all) #om man vill kolla att man får ut rätt gest

Kropp = [11, 10, 1, 0]

H_arm = [1, 3, 5, 7, 9]
V_arm = [1, 2, 4, 6, 8]

H_ben = [11, 12, 14, 16, 18]
V_ben = [11, 13, 15, 17, 19]


for x in range(0,len(gest_all)): #printar alla versioner av gesten
  gest = gest_all[x]

  gest_f1 = gest[0:60]

  x = gest_f1[0::3]
  y = gest_f1[1::3]
  z = gest_f1[2::3]
  #print(x,y,z, sep='\n') #printar kordinaterna

  print(gest[-2])

  ax = plt.axes(projection = '3d')
  ax.scatter(x, y, z)
  ax.plot(x[Kropp], y[Kropp], z[Kropp], label = "Kropp")
  ax.plot(x[H_arm], y[H_arm], z[H_arm], label = "Höger Arm")
  ax.plot(x[V_arm], y[V_arm], z[V_arm], label = "Vänster Arm")
  ax.plot(x[H_ben], y[H_ben], z[H_ben], label = "Höger Ben")
  ax.plot(x[V_ben], y[V_ben], z[V_ben], label = "Vänster Ben")

  ax.view_init(elev=-70, azim=90)
  plt.legend()
  plt.show()

