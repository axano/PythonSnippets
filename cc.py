'''
The goal of this case is training a classifier to detect fraudulent transactions for credit card data.
More information of the case can be found in the pdf on leho.

Firstly we import some packages
'''

# Importing some packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from sklearn.utils import shuffle

'''
The full creditcard dataset is avaiable. 
The data is present in src -> data_ex. 
read in 'creditcard.csv' and check what columns you have and their datatypes.
Print the first 5 rows of the dataframe.
'''
creditcardsDataframe = pd.read_csv('creditcard.csv')
# Default n of head is 5
print(creditcardsDataframe.head())


'''
Drop the two columns 'Index' and 'Time' and store your result. 
Check if you succeeded by displaying the resulting dataframe. 
'''
creditcardsDataframe.reset_index(drop=True, inplace=True)
creditcardsDataframe = creditcardsDataframe.drop(['Time'], axis=1)
print(creditcardsDataframe.head())


'''
This dataset was partly anonimized, only the columns Time and amount are original. 
Looking at the resulting dataframe of the previous codeblock:

Is there any categorical/numerical data ? 

What preprocessing can we do to this dataset ? 

After answering these questions for yourself, 
make a decision for your preprocessing step and execute this on the resulting dataframe of the previous codeblock.

Note : Preprocessing functions from scikit-learn usually return a numpy array.
       You can put the result back into a pandas dataframe and don't forget to add column names. 
'''
# https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/


'''
It's best practice to randomly shuffle the rows of a dataset. Scikit-learn has all sorts of usefull 
preprocessing functions. In the code below the function shuffle is used. Everytime it's called it returns a randomly
shuffle dataset. You can adapt 'df' in the code below if needed.

Note: The shuffle function will return a pandas dataframe if a pandas dataframe is given. 
'''

creditcardsDataframe = shuffle(creditcardsDataframe)

'''
The 'Class' column of the dataframe contains the labels. The value 1 indicates fraud, 0 normal transactions.
This column can be used to filter the dataframe df into two parts. 
Filter the dataset and store them into two dataframes: df_normal and df_fraud.
'''
# true goes to the right df
# https://stackoverflow.com/questions/33742588/pandas-split-dataframe-by-column-value
creditcardsDataframeFraud, creditcardsDataframeNormal = [x for _, x in creditcardsDataframe.groupby(creditcardsDataframe['Class'] < 1)]
print(creditcardsDataframeFraud.head())
print(creditcardsDataframeNormal.head())

'''
We will use a part of the data for training and a part for testing. 
A good split of training and test data is a 70%/30% split.

The first 70% of df_normal and df_fraud should be stored in df_train
The last 30% should be stored in df_test. 

Note : You can use pd.concat() to stick together dataframes.  
'''
### TRAIN
# Takes upper 70%
df_trainNoFraud = creditcardsDataframeNormal.head(int(len(creditcardsDataframeNormal)*(70/100)))
df_trainFraud = creditcardsDataframeFraud.head(int(len(creditcardsDataframeFraud)*(70/100)))
dfTrain = pd.concat([df_trainFraud,df_trainNoFraud])
dfTrain = shuffle(dfTrain)

### TEST
# Takes lower 30%
df_TestNoFraud = creditcardsDataframeNormal.tail(int(len(creditcardsDataframeNormal)*(30/100)))
df_TestFraud = creditcardsDataframeFraud.tail(int(len(creditcardsDataframeFraud)*(30/100)))
dfTest = pd.concat([df_TestNoFraud, df_TestFraud])
dfTest = shuffle(dfTest)


'''
We will use the K-nearest neighbour algorithm. 
See: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

You should try out different values for 'n_neighbors', 'weights', 'p', 'metric'. 
Stick to the rule of thumb to start with simple KNN parameters. Improve accuracy by trying out different parameters. 

The explanation for n_neighbors, weights and metric should be clear from the slides.

The parameter 'p' relates to one specific metric family, the 'minkwoski' metric. 
setting p=1,2,3 gives different metrics if metric='minkowski'. This is the default value for the metric argument

Setting the argument 'n_jobs=-1' will use all of your cpu cores. This gives you faster results but make sure you don't 
have any other heavy applications open while running the code with this option.

So make a KNN object now starting with some simple arguments and store it. 
'''

neigh = KNeighborsClassifier(n_neighbors=3, weights='uniform', p=2, metric='minkowski', n_jobs=-1)

'''
Use the KNN model to train on the df_train dataset. 
The function for training expects two arguments:  a dataframe and labels. 
Make sure the dataframe you pass into the function doesn't contain the column for labels.
'''
dfTrainLabels = dfTrain['Class']
dfTrain = dfTrain.drop(['Class'], axis=1)
print(dfTrain.head())
print(dfTrainLabels.head())
neigh.fit(dfTrain.values, dfTrainLabels)

'''
Using the trained KNN, predict on the test dataset and store the results.
'''
dfTest = dfTest.drop(['Class'], axis=1)
print(dfTest.head())
# Takes 30 s'
prediction = neigh.predict(dfTest.values)
# should return 492 fraud cases
creditcardsDataframe['Class'].value_counts()
print(prediction)
print(type(prediction))
'''
It's time to make a visualization of the results. We will compare the results of the prediction versus the ground 
truth of the labels.

1) Make two arrays of size of test dataset. 
One has values from 0 to size of test dataset. Second array has integer values '1' .

2) Make a boolean array that is true for fraud rows of the test dataset and false otherwise. 

3) Make two scatter plots. A scatter plot of matplotlib expects two numpy arrays. One for the x-values and one for 
   the y-values. For an example of this look at 'src->case->graph.jpg'.
   - Make one scatter plot with points y = 1 and the other with y= 1.5
   - Only the predicted fraud points should be plotted. plot these for y=1 . For the plot with y=1.5 plot points
     with label fraud.
   - Give the predicted fraud points the color red. Give the labeled fraud points the color yellow.
   - give arguments: 'marker', 's (=point size)', 'label'. To both scatter plots.
   - give the plot a title, show the labels and set a ylim such that all points can be seen.

   => show the plot and check if everything is as expected.
'''

plt.plot(prediction)
plt.show()
'''
Calculate the confusion matrix. 
Since we have only two classes for this classifier problem the matrix will be of shape = (2,2). 

Determine 'True Positives', 'True negatives','False positives','False negatives' for this matrix.
Make a dictionary of these four values and store it. display the dictionary.

repeat the process of changing KNN-parameters and producing results to improve your algorithm.
At most there should be 50 misclassified data points. 

Note: The order of the columns can be set for the confusion_matrix. 
      Make sure to set these, because the first value encountered will be set as first columns. 
      This will be random depending on the shuffling of the dataset!
'''

#tn, fp, fn, tp = confusion_matrix(dfTest, prediction).ravel()
confMatrix = confusion_matrix(dfTest, prediction)
plt.plot(confMatrix)
plt.show()

'''
Provide following plots and answer the questions in the comment block below:

- The plots of your best and second best parameter choice. You can save the plot by right-clicking on it
  (see plot_save.jpg).


Questions:
1) Data reading: After dropping the columns 'Index' and 'Time', what columns do we have left?
2) Data preprocessing: We've dropped the column 'Time'. What information do you lose when doing this?
If we keep the 'Time' and shuffle the dataset what information do we lose ?
3) KNN object: Give the parameters that give the best and second best results for you.
4) Prediction: If we train on the entire dataset (with k=1) and tried to make predictions on points within
our dataset, what prediction accuracy would we have ?  
5) copy the printout of the dictionary with confusion matrix values, for your best and second best results.
'''

'''
1) V#, amount and class
2) due to the fact that we keep the time parameter, we could still resort the dataset based on time
3)
4) the accuracy will be less because we will compare every credit card transaction with less neighbours 
5)
'''
