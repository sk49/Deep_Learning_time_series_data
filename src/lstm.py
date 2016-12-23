#!usr/bin/python

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.callbacks import History
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Nadam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime
import sys
sys.stdout = sys.stderr

# configuration configuration
configuration = {
	"input_filename": None,
	"n_layers": None,
	"dropout_fraction_ru": None,
	"dropout_fraction_rw": None,
	"layer_dimensions": None,
	"optimizer": None,
	"learning_rate": None,
	"momentum": None,
	"training_percent": None,
	"err_metric": None,
	"output_filename": None,
	"logfile": None,
	"epoch": None,
}

'''
Setting configuration

Default values,
    n_layers = 3
    dropout_fraction_ru = 0
    dropout_fraction_rw = 0
    layer_dimensions = [1,4,1]
    optimizer = adam
    learning_rate = 0.0
    momentum = 0.0
    err_metric = mean_squared_error
    logfile = [output_filename]_log
    epoch = 100

Required values,
    input_filename
    output_filename
'''
def set_configuration(input_filename, output_filename, n_layers=3, dropout_fraction_ru=0, dropout_fraction_rw=0, layer_dimensions=[1,4,1], optimizer="adam", learning_rate=0.0, momentum=0.0, training_percent=0.7, err_metric="mean_squared_error", logfile=None, epoch=100):
    # Validation for input_filename and output_filename
    if not input_filename:
        raise ValueError("Invalid argument: input_filename")
    elif not output_filename:
        raise ValueError("Invalid argument: output_filename")
    # setting the logfile
    if not logfile:
		logfile = output_filename+"_log"

    # Validation of training and test percent
    if training_percent == 0.0 or training_percent > 1.0:
        raise ValueError("Invalid argument: training_percent should be in range (0.0, 1.0]")

    configuration["input_filename"] = input_filename
    configuration["n_layers"] = n_layers
    configuration["layer_dimensions"] = layer_dimensions
    configuration["dropout_fraction_ru"] = dropout_fraction_ru
    configuration["dropout_fraction_rw"] = dropout_fraction_rw
    configuration["optimizer"] = optimizer
    configuration["learning_rate"] = learning_rate
    configuration["momentum"] = momentum
    configuration["training_percent"] = training_percent
    configuration["err_metric"] = err_metric
    configuration["output_filename"] = output_filename
    configuration["logfile"] = logfile
    configuration["epoch"] = epoch

'''
Create and add the input, hidden and output layers, to a given model.

The number of layers and their dimensions are taken from the configuration
If dropout_fraction_rw is not default (meaning that it has been specified by the user) then we pick that instead of dropout_fraction_ru
Else, we pick dropout_fraction_ru instead.
'''
def add_layers(layer_dimensions, model):
    # Validation
    if not model:
        raise ValueError("Invalid argument: model")

    if configuration["dropout_fraction_rw"] != 0:
	print "Using DROPUT_FRACTION_RW"
	model.add(LSTM(input_dim=layer_dimensions[0], output_dim=layer_dimensions[1], return_sequences=True, dropout_W=configuration["dropout_fraction_rw"]))
        i = 0
        for i in range(1, len(layer_dimensions)-2):
            model.add(LSTM(input_dim=layer_dimensions[i], output_dim=layer_dimensions[i+1], return_sequences=True, dropout_W=configuration["dropout_fraction_ru"]))
        model.add(LSTM(input_dim=layer_dimensions[i], output_dim=layer_dimensions[i+1], return_sequences=False, dropout_W=configuration["dropout_fraction_ru"]))
        model.add(Dense(output_dim=layer_dimensions[len(layer_dimensions)-1]))
    else:
        model.add(LSTM(input_dim=layer_dimensions[0], output_dim=layer_dimensions[1], return_sequences=True, dropout_U=configuration["dropout_fraction_ru"]))
        i = 0
        for i in range(1, len(layer_dimensions)-2):
            model.add(LSTM(input_dim=layer_dimensions[i], output_dim=layer_dimensions[i+1], return_sequences=True, dropout_U=configuration["dropout_fraction_ru"]))
        model.add(LSTM(input_dim=layer_dimensions[i], output_dim=layer_dimensions[i+1], return_sequences=False, dropout_U=configuration["dropout_fraction_ru"]))
        model.add(Dense(output_dim=layer_dimensions[len(layer_dimensions)-1]))

# Convert an array of values into a dataset
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
	a = dataset[i:(i+look_back), 0]
	dataX.append(a)
	dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

'''
Load dataset, normalize it and generate training and test data

If improper file then the exception raised by pandas is thrown
'''
def load_normalize_and_generate_test_and_training_data():
    # fix random seed for reproducibility
    numpy.random.seed(7)

    # load the dataset
    dataframe = None
    try:
        dataframe = pandas.read_csv(configuration["input_filename"], usecols=[1], engine='python', skipfooter=3)
    except Exception, e:
        raise Exception("Error in reading %s: %s", configuration["input_filename"], e)
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    training_data_size = int(len(dataset) * configuration["training_percent"])
    test_data_size = len(dataset) - training_data_size
    training_data, test_data = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(training_data, look_back)
    testX, testY = create_dataset(test_data, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return dataset, trainX, trainY, testX, testY, scaler

# Make predictions
def predict(trainX, trainY, testX, testY, model, scaler):
    train_data_prediction = model.predict(trainX)
    test_data_prediction = model.predict(testX)

    # invert predictions
    train_data_prediction = scaler.inverse_transform(train_data_prediction)
    trainY = scaler.inverse_transform([trainY])
    test_data_prediction = scaler.inverse_transform(test_data_prediction)
    testY = scaler.inverse_transform([testY])
    return train_data_prediction, test_data_prediction, trainY, testY

# Evluation
def evaluate(train_data_prediction, trainY, test_data_prediction, testY, dataset):
    # calculate root mean squared error
    trainScore, testScore = None, None
    if configuration["err_metric"] == "mean_squared_error":
        trainScore = math.sqrt(mean_squared_error(trainY[0], train_data_prediction[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], test_data_prediction[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

    return trainScore, testScore

# Plot the training and test data
def plot_and_save(dataset, train_data_prediction, test_data_prediction, scaler, look_back=1):
    # shift train predictions for plotting
    train_prediction_plot_data = numpy.empty_like(dataset)
    train_prediction_plot_data[:, :] = numpy.nan
    train_prediction_plot_data[look_back:len(train_data_prediction)+look_back, :] = train_data_prediction

    # shift test predictions for plotting
    test_prediction_plot_data = numpy.empty_like(dataset)
    test_prediction_plot_data[:, :] = numpy.nan
    test_prediction_plot_data[len(train_data_prediction)+(look_back*2)+1:len(dataset)-1, :] = test_data_prediction

    plt.plot(scaler.inverse_transform(dataset), label="dataset")
    plt.plot(train_prediction_plot_data, label="train prediction")
    plt.plot(test_prediction_plot_data, label="test prediction")
    legend = plt.legend(fontsize=17,loc='upper center', bbox_to_anchor=(0.47, -0.04),fancybox=False, shadow=False, ncol=5)
    plt.savefig(configuration["output_filename"]+".png", bbox_extra_artists=(legend,), bbox_inches='tight')
    plt.show()
    plt.close()

'''
Add log corresponding to the given key, to the logfile
'''
def add_log(logfile, log_type, value):
	file_obj = open(logfile, 'a')
	file_obj.write(log_type+" = "+str(value)+"\n")
	file_obj.close()

'''
Run the neural network

Takes the configuration present in the configuration
If the configuration was not set, then an exception is raised.
'''
def run():
    dataset, train_data, test_data = None, None, None
    try:
        dataset, trainX, trainY, testX, testY, scaler = load_normalize_and_generate_test_and_training_data()
    except Exception as e:
        raise e

    # create and fit the LSTM network. (Also calculating time taken for the same)
    start = datetime.datetime.now()
    model = Sequential()
    # adding layers
    try:
        add_layers(configuration["layer_dimensions"], model)
    except ValueError as vs:
        raise v
    opt = None
    if configuration["optimizer"] == "adam":
	opt = Adam(lr=configuration["learning_rate"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    elif configuration["optimizer"] == "nadam":
	opt = Nadam(lr=configuration["learning_rate"])
    elif configuration["optimizer"] == "SGD":
	opt = SGD(lr=configuration["learning_rate"], momentum=configuration["momentum"])
    elif configuration["optimizer"] == "RMSprop":
	opt = RMSprop(lr=configuration["learning_rate"])
    # model.compile(loss=configuration["err_metric"], optimizer=configuration["optimizer"])
    model.compile(loss=configuration["err_metric"], optimizer=opt)
    # Validation is set to 30%
    history = History()
    model.fit(trainX, trainY, nb_epoch=configuration["epoch"], batch_size=1, verbose=2, validation_split=0.3, callbacks=[history])
    training_time_in_seconds = (datetime.datetime.now() - start).total_seconds()

    # predict
    train_data_prediction, test_data_prediction, trainY, testY = predict(trainX, trainY, testX, testY, model, scaler)
    #evaluate
    trainScore, testScore = evaluate(train_data_prediction, trainY, test_data_prediction, testY, dataset)
    # need to log the training time to logfile
    print "Training Time: ", training_time_in_seconds
    add_log(configuration["logfile"], "train-time", training_time_in_seconds)
    # need to log the trainScore and testScore to the logfile
    add_log(configuration["logfile"], "train-RMSE", trainScore)
    add_log(configuration["logfile"], "test-RMSE", testScore)
    print "Final Cross-Validation result: ", history.history["val_loss"][-1]
    add_log(configuration["logfile"], "Final Cross-validation result", history.history["val_loss"][-1])
    # plot and save figure
    plot_and_save(dataset, train_data_prediction, test_data_prediction, scaler)

'''
Append run configuration to run_configs file
'''
def append_config():
    file_obj = open(configuration["logfile"], 'a')
    for k, v in configuration.items():
	file_obj.write(k+"="+str(v)+"\n")
    file_obj.write("\n\n")
    file_obj.close()
