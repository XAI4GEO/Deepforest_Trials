import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import tensorflow as tf
from keras import backend as k
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def _create_model(nneurons, nfilters, ndropout, npool):
    inputs = keras.Input((100, 100, 3))
    x = keras.layers.Conv2D(nneurons[0], (nfilters[0], nfilters[0]), padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(npool[0], npool[0]), data_format='channels_last')(x)
    x = keras.layers.Dropout(ndropout[0])(x)

    x = keras.layers.Conv2D(nneurons[1], (nfilters[1], nfilters[1]), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[1], npool[1]), data_format='channels_last')(x)
    x = keras.layers.Dropout(ndropout[0])(x)

    x = keras.layers.Conv2D(nneurons[2], (nfilters[2], nfilters[2]), padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(npool[2], npool[2]), data_format='channels_last')(x)
    x = keras.layers.Dropout(ndropout[0])(x)

    pooledOutput = keras.layers.GlobalAveragePooling2D()(x)
    pooledOutput = keras.layers.Dense(nneurons[3])(pooledOutput)
    outputs = keras.layers.Dense(nneurons[4])(pooledOutput)

    model = keras.Model(inputs, outputs)
    return model

def _euclidean_distance(vectors):
    (featA, featB) = vectors
    sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
    return k.sqrt(k.maximum(sum_squared, k.epsilon()))

def siamese_model(nneurons, nfilters, ndropout, npool):
    feature_extractor_model = _create_model(nneurons, nfilters, ndropout, npool)
    imgA = keras.Input(shape=(100, 100, 3))
    imgB = keras.Input(shape=(100, 100, 3))
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

def load_model_weights(model, weights):
    model.load_weights(weights)    

def read_pp_data(filename):
    data = np.load(filename)
    return data['X'].transpose(0,1,3,4,2), data['y']

def generate_train_image_pairs(X):
    
    pair_images = []
    pair_labels = []
    pair_index = []

    for labeli in range(X.shape[0]):
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                for labelj in range(X.shape[0]):
                    image = X[labeli,i,:,:,:]
                    image2 = X[labelj,j,:,:,:]
                    pair_images.append((image, image2))
                    if labeli == labelj:
                        pair_labels.append(1)
                    else:
                        pair_labels.append(0)
    return np.array(pair_images), np.array(pair_labels)

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, start_from_epoch=20, restore_best_weights=True)
    history = model.fit([X_train[:, 0], X_train[:, 1]], y_train[:], validation_data=([X_val[:,0], X_val[:,1]], y_val[:]), batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    return history

def predict_model(model, images_pair):
    labels_pred = model.predict([images_pair[:, 0], images_pair[:, 1]])
    return labels_pred

def plots(history, labels_pred, labels_pair, filename):
    _plot_history(history, ['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
    _plot_labels(labels_pred, labels_pair)
    _plot_histogram(labels_pred, labels_pair)
    save_image(filename) 

def _plot_history(history, metrics):
    """
    Plot the training history

    Args:
        history (keras History object that is returned by model.fit())
        metrics (str, list): Metric or a list of metrics to plot
    """
    fig1 = plt.figure()
    history_df = pd.DataFrame.from_dict(history.history)
    plt.plot(history_df[metrics], label=metrics)
    plt.xlabel("epochs")
    plt.ylabel("metric")
    plt.legend()
    plt.ylim(0, 1)

def _plot_labels(labels_pred, labels_pair):
    fig2 = plt.figure(figsize=(10, 5))
    sub = fig2.add_subplot(1, 2, 1)
    mask = np.isin(labels_pair, 0)
    sub.plot(labels_pred[mask])
    sub.plot(labels_pair[mask])
    sub.set_xlabel('Epochs')
    sub.set_ylabel('Similarity Value')
    sub.set_title('Dissimilar Pairs')

    sub = fig2.add_subplot(1, 2, 2)
    mask = np.isin(labels_pair, 1)
    sub.plot(labels_pred[mask])
    sub.plot(labels_pair[mask])
    sub.set_xlabel('Epochs')
    sub.set_ylabel('Similarity Value')
    sub.set_title('Similar Pairs')

def _plot_histogram(labels_pred, labels_pair):
    fig3 = plt.figure(figsize=(10, 5))

    mask = np.isin(labels_pair, 0)
    counts, bins = np.histogram(labels_pred[mask])
    sub = fig3.add_subplot(1, 2, 1)
    sub.stairs(counts, bins)
    sub.set_xlabel('Similarity Value')
    sub.set_ylabel('Count')
    sub.set_title('Dissimilar Pairs')

    mask = np.isin(labels_pair, 1)
    sub = fig3.add_subplot(1, 2, 2)
    counts, bins = np.histogram(labels_pred[mask])
    sub.stairs(counts, bins)
    sub.set_xlabel('Similarity Value')
    sub.set_ylabel('Count')
    sub.set_title('Similar Pairs')

def save_image(filename):
    
    # PdfPages is a wrapper around pdf 
    # file so there is no clash and
    # create files with no error.
    p = PdfPages(filename)
      
    # get_fignums Return list of existing
    # figure numbers
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
      
    # iterating over the numbers in list
    for fig in figs: 
        
        # and saving the files
        fig.savefig(p, format='pdf') 
          
    # close the object
    p.close()  

def main():

    keras.backend.set_image_data_format('channels_last')

    #Base parameters
    filename = 'trial.pdf'
    nneurons = [32, 64, 96, 64, 32]
    nfilters = [5, 5, 5]
    ndropout = [0.4, 0.4, 0.4]
    npool = [2, 2, 2]
    lr = 0.001 
    batchsize = 64
    epochs = 10
    print ('Created Base Parameters')

    #Load data
    X, y = read_pp_data('train_data_da.npz')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
    print ('Loaded Data')

    #Create model
    model = siamese_model(nneurons, nfilters, ndropout, npool)
    # load_model_weights(model, 'weights_base.h5')
    # print ('Created Model')

    #Compile model
    metrics = [keras.metrics.BinaryAccuracy(threshold=0.5)]
    compile_model(model, lr, metrics)
    # print ('Compiled Model')

    #Train data
    history = train_model(model, X_train, y_train, X_val, y_val, batchsize, epochs)
    # print ('Trained Model')

    #Model Prediction
    images_pair = np.append(X_train, X_val, axis = 0)
    labels_pred = predict_model(model, images_pair)
    labels_pair = np.append(y_train, y_val, axis = 0)
    print ('Model Prediction Completed')

    #Plot data
    plots(history, labels_pred, labels_pair, filename)
    print ('Created Plots')

    print (history.history)
    print ('Best Accuracy on Validation Set: ',max(history.history['val_binary_accuracy']))

    model.save_weights('optimized_weights.h5')

if __name__ == "__main__":
    main()
