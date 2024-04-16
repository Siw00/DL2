#Experiment 2: Python Program for Data Visualization

import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#PART 1
# add only if using colab 
#%matplotlib inline
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])
plt.plot(xpoints, ypoints)
plt.show()


plt.savefig("mygraph.png")
sys.stdout.flush()

# PART 2
# add only if using colab 
#%matplotlib inline
x = np.array(["A","B","C","D"])
y = np.array([3, 8, 1, 10])
plt.bar(x,y)
plt.show()
plt.savefig("mygraph.png")
sys.stdout.flush()


# PART 3
# add only if using colab 
#%matplotlib inline
y = np.array([35, 25, 25, 15])
mylabels = ["Apples","Bananas","Cherries","Dates"]
plt.pie(y, labels = mylabels)
plt.show()
plt.savefig("mygraph.png")
sys.stdout.flush()


# Experiment 3: Data Pre-processing using Linear Regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Creating DataFrame
df = pd.DataFrame({"Job Position": ['CEO', 'Senior Manager', 'Junior Manager', 'Employee', 'Assistant Staff'], 
                   "Years of Experience": [5, 4, 3, None, 1], 
                   "Salary": [100000, 80000, None, 40000, 20000]})

# Dropping rows with missing values
dropped_df = df.drop([2, 3], axis=0)
print("After removing the missing values")
print(dropped_df)

# Dropping rows with missing values for training
train_df = df.drop([2, 3], axis=0)

# Creating linear regression model for predicting salary based on experience
regr_salary = LinearRegression()
regr_salary.fit(train_df[['Years of Experience']], train_df[['Salary']])

# Predicting salary for 3 years of experience
print("Salary with 3 years of experience:")
print(regr_salary.predict([[3]]))

# Creating linear regression model for predicting experience based on salary
regr_experience = LinearRegression()
regr_experience.fit(train_df[['Salary']], train_df[['Years of Experience']])

# Predicting years of experience for salary 40000.0
print("Years of experience with Salary 40000.0:")
print(regr_experience.predict([[40000.0]]))


#Experiment 4: Implementation of k-Nearest Neighbor Algorithm for Iris Dataset  Classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# import the iris dataset
iris = datasets.load_iris()
# print(iris)

X = iris.data
# print(X)
y = iris.target

# splitting X and y into training and testing sets in 80:20 Ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

k_range = range(1, 11)
score = {}
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    score[k] = accuracy_score(y_test, knn_pred)
    scores_list.append(accuracy_score(y_test, knn_pred))

plt.plot(k_range, scores_list)
plt.xlabel("Value of K for kNN")
plt.ylabel("Testing Accuracy")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("Final Accuracy:", accuracy_score(y_test, knn_pred))

classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
x_new = [[3, 4, 5, 6], [5, 4, 4, 4]]
y_predict = knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])


#Experiment 6: Predicting Rain Tomorrow with Machine Learning
# for loading the austin_weather.csv file https://github.com/elmoallistair/datacamp-data-analyst-with-python/blob/master/05_introduction-to-data-visualization-with-matplotlib/datasets/austin_weather.csv
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("austin_weather.csv")
data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches', 'SeaLevelPressureLowInches'], axis=1)
data = data.replace('T', 0.0)
data = data.replace('-', 0.0)

# save the data in a csv file
data.to_csv('austin_final.csv')

data = pd.read_csv("austin_final.csv")
X = data.drop(['PrecipitationSumInches'], axis=1)
Y = data['PrecipitationSumInches']
Y = Y.values.reshape(-1, 1)

day_index = 798  # consider a random day in the dataset
days = [i for i in range(Y.size)]

clf = LinearRegression()
clf.fit(X, Y)

inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45],
                 [57], [29.68], [10], [7], [2], [0], [20], [4], [31]])

inp = inp.reshape(1, -1)

# print the output.
print('The precipitation in inches for the input is:', clf.predict(inp))

print("the precipitation trend graph:")
plt.scatter(days, Y, color='b')
plt.scatter(days[day_index], Y[day_index], color='r', marker='s')
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabel("Precipitation in inches")
plt.show()

x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                  'SeaLevelPressureAvgInches', 'VisibilityAvgMiles', 'WindAvgMPH'], axis=1)

# Precipitation vs selected attributes graph:
print("Precipitation vs selected attributes graph:")
for i in range(x_vis.columns.size):
    plt.subplot(3, 2, i + 1)
    plt.scatter(days[:100], x_vis[x_vis.columns.values[i]][:100], color='b', marker=".")
    plt.scatter(days[day_index], x_vis[x_vis.columns.values[i]][day_index], color='r', marker="s")
    plt.title(x_vis.columns.values[i])

plt.show()

#Experiment 10: Handwritten Digit Recognition Using Neural Network 
from keras.datasets import mnist
from matplotlib import pyplot

# Load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()

# Summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))

# Plot first few images
for i in range(9):
    # Define subplot
    pyplot.subplot(330 + 1 + i)
    # Plot raw pixel data
    pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))

# Show the figure
pyplot.show()

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)  # (60000, 28, 28) -- 60000 images, each 28x28 pixels
print("Test data shape", x_test.shape)  # (10000, 28, 28) -- 10000 images, each 28x28

# Flatten the images
image_vector_size = 28 * 28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

from keras.layers import Dense
from keras.models import Sequential

image_size = 784  # 28*28
num_classes = 10  # ten unique digits

model = Sequential()

# HERE THE HIDDEN LAYER HAS 500 NODES
model.add(Dense(units=500, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

# HERE THE HIDDEN LAYER HAS 1000 NODES
model = Sequential()
model.add(Dense(units=1000, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

# HERE THE HIDDEN LAYER HAS 20000 NODES
model = Sequential()
model.add(Dense(units=2000, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()

#Experiment 12: Handwritten Digits Recognition using Convolutional Neural Network  (CNN)
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape input data to match CNN input shape (add channel dimension)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encode the target labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define CNN architecture
cnn = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Output layer with 10 neurons for 10 classes
])

# Compile the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = cnn.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)


#Experiment 13: Handwritten Digits Recognition using Recurrent Neural Network (RNN)
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape input data to be 3D for RNN input (samples, time steps, features)
time_steps = x_train.shape[1]  # Number of time steps
features = x_train.shape[2]     # Number of features (pixels)
x_train = x_train.reshape(x_train.shape[0], time_steps, features)
x_test = x_test.reshape(x_test.shape[0], time_steps, features)

# One-hot encode the target labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define RNN architecture
rnn = Sequential([
    LSTM(128, input_shape=(time_steps, features)),
    Dense(10, activation='softmax')  # Output layer with 10 neurons for 10 classes
])

# Compile the model
rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
rnn.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = rnn.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

#Experiment 14: Sentiment Analysis on Movie Reviews using LSTM
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing import sequence

# Load the IMDb dataset
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# Pad sequences to ensure uniform length for input to LSTM
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Define the LSTM model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=3, batch_size=64)

# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
