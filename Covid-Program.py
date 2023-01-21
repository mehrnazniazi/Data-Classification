"""
This code is written by Mehrnaz Niazi and it loads a CSV file containing COVID-19 data. The dataset containing 11000 data points which can be downloaded from "https://www.kaggle.com/datasets/meirnizri/covid19-dataset" provided on Kaggle. The code performs preprocessing on the data by mapping certain values. Then it selects the top 10 features using the SelectKBest method and visualizes the counts of each feature. It then one-hot encodes the target column and splits the data into training and test sets. The code defines three different functions to train and evaluate three different types of neural networks: a Multi-Layer Perceptron (MLP), a Convolutional Neural Network (CNN), and a Recurrent Neural Network (RNN). It then calls each of these functions to train and evaluate the corresponding neural network on the data.
"""

from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GRU, BatchNormalization
from keras.models import Sequential
import numpy as np
import seaborn as sns
import numpy as np

# Load the train and test data
data = pd.read_csv('/content/Covid Data.csv')

# Create a new column that indicates whether the date is "9999-99-99" or not
data['DATE_DIED'] = data['DATE_DIED'].map(lambda x: 0 if x == '9999-99-99' else 1)


# map 97,98,99 to 3,4,5
col=data.columns
for i in col:
  data[i]= data[i].map(lambda x: 3 if x == 97 else x)
  data[i]= data[i].map(lambda x: 4 if x == 98 else x)
  data[i]= data[i].map(lambda x: 5 if x == 99 else x)
  
# Extract the target column from the data
Y_data = data['CLASIFFICATION_FINAL']

# Extract the feature columns from the data
X_data = data.drop('CLASIFFICATION_FINAL',axis=1)

# Initialize the SelectKBest model with the number of features you want to select
selector = SelectKBest(f_regression, k=10)
# Fit the SelectKBest model to the data
selector.fit(X_data, Y_data)
# Get the selected features
selected_features = X_data.columns[selector.get_support()]
# Create a new data set with only the selected features
X_data = X_data[selected_features]


cols = X_data.columns
for i in cols:
    print(f"{i} : {X_data[i].value_counts()}")
    sns.countplot(x=X_data[i])
    plt.title(i)
    plt.show()

Y_data = pd.get_dummies(Y_data)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=60)


def train_mlp(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)
    return history,model

def train_cnn(X_train, y_train, X_test, y_test):
      model = Sequential()
      model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
      model.add(MaxPooling1D(pool_size=2))
      model.add(Dropout(0.5))
      model.add(Flatten())
      model.add(Dense(7, activation='softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
      history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)
      return history,model

def train_rnn(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(GRU(16, input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)
    return history,model


# train the models 
mlp_history,mlp_model = train_mlp(X_train, y_train, X_test, y_test)
cnn_history,cnn_model  = train_cnn(X_train, y_train, X_test, y_test)
rnn_history,rnn_model = train_rnn(X_train, y_train, X_test, y_test)


# Plot the accuracy and mean squared error for each model
models = ['MLP',  'CNN', 'RNN']
histories = [mlp_history, cnn_history, rnn_history]

# Set up the figure
fig, ax = plt.subplots(figsize=(12,8))

# Plot the accuracy values
for history in histories:
  ax.plot(history.history['accuracy'])
ax.set_title('Accuracy')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.legend(['MLP', 'CNN', 'RNN'], loc='upper left')

# Show the plot
plt.show()


# Set up the figure
fig, ax = plt.subplots(figsize=(12,8))

# Plot the test accuracy values
mlp_test_acc = mlp_model.evaluate(X_test, y_test, verbose=0)[1]
cnn_test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)[1]
rnn_test_acc = rnn_model.evaluate(X_test, y_test, verbose=0)[1]

ax.bar(['MLP', 'CNN', 'RNN'], [mlp_test_acc, cnn_test_acc, rnn_test_acc])
ax.set_title('Test Accuracy')
ax.set_ylabel('Accuracy')

# Add the accuracy numbers above the bars
for i, v in enumerate([mlp_test_acc, cnn_test_acc, rnn_test_acc]):
  ax.text(i, v + 0.01, str(round(v*100,2)) + '%', color='blue')


# Show the plot
plt.show()


# Create a dictionary with the test accuracy for each model
test_acc = {'MLP': mlp_test_acc, 'CNN': cnn_test_acc, 'RNN': rnn_test_acc}

# Convert the dictionary to a dataframe
test_acc_df = pd.DataFrame(list(test_acc.items()), columns=['Model', 'Test Accuracy'])
test_acc_df['Test Accuracy'] = test_acc_df['Test Accuracy'].round(4)*100

# Display the dataframe
print(test_acc_df)