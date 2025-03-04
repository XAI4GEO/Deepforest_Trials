# Import libraries
import keras
import tensorflow as tf
import os
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
from pathlib import Path
import xarray as xr
import json
import matplotlib.pyplot as plt
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, Callback

# Set Keras backend for image data format
keras.backend.set_image_data_format("channels_last")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Model creation
def _create_model(nneurons, nfilters, ndropout, npool):
    # 4 layers
    inputs = keras.Input((400, 400, 3))
    
    x = keras.layers.Conv2D(nneurons[0], (nfilters[0], nfilters[0]), padding="same", activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[0], npool[0]), data_format='channels_last')(x)
    #x = keras.layers.Dropout(ndropout[0])(x)

    x = keras.layers.Conv2D(nneurons[1], (nfilters[1], nfilters[1]), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[1], npool[1]), data_format='channels_last')(x)
    #x = keras.layers.Dropout(ndropout[0])(x)

    x = keras.layers.Conv2D(nneurons[2], (nfilters[2], nfilters[2]), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[2], npool[2]), data_format='channels_last')(x)
    #x = keras.layers.Dropout(ndropout[0])(x)

    x = keras.layers.Conv2D(nneurons[3], (nfilters[3], nfilters[3]), padding="same", activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[3], npool[3]), data_format='channels_last')(x)
    x = keras.layers.Dropout(ndropout[0])(x)

    pooledOutput = keras.layers.GlobalAveragePooling2D()(x)
    pooledOutput = keras.layers.Dense(nneurons[4])(pooledOutput)
    outputs = keras.layers.Dense(nneurons[5])(pooledOutput)

    model = keras.Model(inputs, outputs)
    return model

def _euclidean_distance(vectors):
    (featA, featB) = vectors
    squared = k.square(featA-featB)
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return squared #k.sqrt(k.maximum(sum_squared, k.epsilon()))

def siamese_model(nneurons, nfilters, ndropout, npool):
    feature_extractor_model = _create_model(nneurons, nfilters, ndropout, npool)
    imgA = keras.Input(shape=(400, 400, 3))
    imgB = keras.Input(shape=(400, 400, 3))
    featA = feature_extractor_model(imgA)
    featB = feature_extractor_model(imgB)
    distance = keras.layers.Lambda(_euclidean_distance)([featA, featB])
    outputs = keras.layers.Dense(1, activation="sigmoid")(distance)
    model = keras.Model(inputs=[imgA, imgB], outputs=outputs)
    return model

def compile_model(model, lr, metrics):
    opt = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    metrics = metrics
    model.compile(loss=loss, optimizer=opt, metrics=metrics)


def plot_history(history, metrics):
    """
    Plot the training history

    Args:
        history (keras History object that is returned by model.fit())
        metrics (str, list): Metric or a list of metrics to plot
    """
    history_df = pd.DataFrame.from_dict(history.history)
    fig = plt.figure(figsize=(10, 5))
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("metric")
    plt.ylim(0, 1)


def plot_prediction(labels_pred, labels_pair):
    fig = plt.figure(figsize=(10, 5))

    sub = fig.add_subplot(1, 2, 1)
    sub.plot(labels_pred[::2])
    sub.plot(labels_pair[::2])

    sub = fig.add_subplot(1, 2, 2)
    sub.plot(labels_pred[1::2])
    sub.plot(labels_pair[1::2])


def plot_prediction_hist(labels_pred):
    fig = plt.figure(figsize=(10, 5))

    counts, bins = np.histogram(labels_pred[::2])
    sub = fig.add_subplot(1, 2, 1)
    sub.stairs(counts, bins)

    sub = fig.add_subplot(1, 2, 2)
    counts, bins = np.histogram(labels_pred[1::2])
    sub.stairs(counts, bins)

def lr_schedule(epoch):
        if epoch<=10:
            return 0.001
        elif epoch<=35:
            return 0.0002
        elif epoch<=60:
            return 0.00005
        else:
            return 0.00001
        
# Custom callback to plot predictions every 10 epochs
class PlotPredictionsCallback(Callback):
    def __init__(self, x_val, y_val, dir_training):
        super(PlotPredictionsCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.dir_training = dir_training
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 2 == 0:
            
            # Make predictions on the validation data
            images_pair = self.x_val
            labels_pred = self.model.predict([images_pair[:,0], images_pair[:,1]])
            
            dir_epoch = self.dir_training / f'epoch_{epoch+1}'
            dir_epoch.mkdir(exist_ok=True)
            
            # Plots
            plot_prediction(labels_pred, self.y_val)
            plt.savefig(dir_epoch/'prediction.png')
            plot_prediction_hist(labels_pred)
            plt.savefig(dir_epoch/'prediction_hist.png')
            plt.close('all')

# class LoadHistoryCallback(Callback):
#     def __init__(self, pickle_file):
#         super(LoadHistoryCallback, self).__init__()
#         self.pickle_file = pickle_file

#     def on_train_begin(self, logs=None):
#         with open(self.pickle_file, 'rb') as f:
#             self.history = pickle.load(f)

#     def on_epoch_end(self, epoch, logs=None):
#         for key in self.history.history.keys():
#             if key in logs:
#                 logs[key] = self.history.history[key][:]

            
if __name__ == "__main__":
    
    # Configuration
    train_name = 'training_historical_best_cutout200_extend'
    nneurons = [32, 64, 64, 128, 256, 128]
    nfilters = [3, 3, 5, 5]
    ndropout = [0.4]
    npool = [4, 4, 4, 4]
    lr_init = 1.0e-05 # initial learning rate
    
    # Load training data
    # data = xr.open_zarr("../../../data/foresttree/training_data.zarr/")
    data = xr.open_zarr("../../../data/foresttree/training_data_larger_than_200.zarr/")
    images_pair = data["X"].to_numpy()
    labels_pair = data["Y"].to_numpy()
    
    # Scale to [0, 1]
    images_pair = images_pair/255.
    
    # #Create model
    # model = siamese_model(nneurons, nfilters, ndropout, npool)
    results_dir = Path('./results_training_slurm/best_so_far/training_historical_best_cutout200/')
    f_weights = results_dir/'optimized_weights.h5'
    f_history = results_dir/'history.pkl'
    with open(f_history, 'rb') as file_pi:
        history = pickle.load(file_pi)
        model = history.model
    # load weight
    model.load_weights(f_weights)
    
    
    #Compile model
    metrics = [keras.metrics.BinaryAccuracy(threshold=0.5)]    
    opt = keras.optimizers.Adam(learning_rate=lr_init)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    
    # Configure training    
    dir_training = Path("./results_training_slurm/") / train_name
    dir_training.mkdir(exist_ok=True)
    
    # Set callbacks
    # learning_rate_scheduler = LearningRateScheduler(lr_schedule)
    learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    # load_history_callback = LoadHistoryCallback(f_history)
    plot_pred_callback = PlotPredictionsCallback(images_pair, labels_pair, dir_training)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=20, restore_best_weights=True)
    callbacks = [learning_rate_scheduler, plot_pred_callback, earlystop]
    
    history = model.fit([images_pair[:,0], images_pair[:,1]], labels_pair[:], batch_size=32, epochs=100, validation_split=0.2, callbacks=callbacks)
   
    # Save training results
    metadata ={ 
        "nneurons": nneurons,
        "nfilters": nfilters,
        "ndropout": ndropout,
        "npool":npool,
    } 
    with open(dir_training /"hyperparams.json", "w") as file:
        json.dump(metadata, file)
 
    model.save_weights(dir_training / "optimized_weights.h5")
    with open(dir_training / "history.pkl", "wb") as file_pi:
        pickle.dump(history, file_pi)
    
    # Model evaluation plots
    labels_pred = model.predict([images_pair[:, 0], images_pair[:, 1]])
    plot_history(history, ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
    plt.savefig(dir_training / "history.png")
    plot_prediction(labels_pred, labels_pair)
    plt.savefig(dir_training / "prediction.png")
    plot_prediction_hist(labels_pred)
    plt.savefig(dir_training / "prediction_hist.png")
