import numpy as np
import sys
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import matplotlib.pyplot as plt
import utils
import pickle
from keras.preprocessing.sequence import pad_sequences

# Performs classification using CNN.

FREQ_DIST_FILE = './train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = './train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = './train-processed.csv'
TEST_PROCESSED_FILE = './test_new-processed.csv'
GLOVE_FILE = '../dataset/glove-seeds.txt'
CLASSIFIER = 'SVM'
MODEL_FILE = 'cnn-feats-%s.pkl' % CLASSIFIER
C = 0.1
MAX_ITER = 1000
dim = 200


def get_glove_vectors(vocab):
    """
    Extracts glove vectors from seed file only for words present in vocab.
    """
    print ('Looking for GLOVE seeds')
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r', encoding="utf8") as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.strip().split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print ('\n')
    return glove_vectors


def get_feature_vector(tweet):
    """
    Generates a feature vector for each tweet where each word is
    represented by integer index based on rank in vocabulary.
    """
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=True):
    """
    Generates training X, y pairs.
    """
    tweets = []
    labels = []
    print ('Generating feature vectors')
    with open(csv_file, 'r', encoding="utf8") as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
                labels.append(int(sentiment))
            utils.write_status(i + 1, total)
    print ('\n')
    return tweets, np.array(labels)


if __name__ == '__main__':
    train = len(sys.argv) == 1
    np.random.seed(1337)
    vocab_size = 90000
    batch_size = 500
    max_length = 40
    filters = 600
    kernel_size = 3
    vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
    glove_vectors = get_glove_vectors(vocab)
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    # Create and embedding matrix
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
    # Seed it with GloVe vectors
    for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    shuffled_indices = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]
    #balancing dataset
    #tweets, labels = RandomUnderSampler(ratio='majority').fit_sample(tweets, labels)
    #split dataset
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.1, random_state=42)
    if train:
        model = Sequential()
        model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))
        model.add(Dropout(0.4))
        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(600))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #filepath = "./models/4cnn-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5"
        filepath = "./models/4cnn-best_model.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1, save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
        history = model.fit(X_train, y_train, batch_size=128, epochs=16, validation_data=(X_test,y_test), callbacks=[checkpoint, reduce_lr])
        print("Testing...")
        score = model.evaluate(X_test, y_test, verbose=0)
        predicted = model.predict(X_test, batch_size=1024, verbose=1)
        predicted = np.round(predicted[:, 0]).astype(float)
        print('CNN Accuracy score: %.2f %%' % (score[1]*100))
        print("CNN Precision: %.2f%%" % (metrics.precision_score(y_test, predicted, average='weighted') * 100.0))
        print("CNN Recall: %.2f%%" % (metrics.recall_score(y_test, predicted, average='weighted') * 100.0))
        print("CNN F1-Score: %.2f%%" % (metrics.f1_score(y_test, predicted, average='weighted') * 100.0))
        print(metrics.classification_report(y_test, predicted, digits=4))
        #extract CNN feature
        svm_feats = model.predict(tweets, batch_size=128, verbose=1)
        #spliting data
        svm_x_train, svm_x_val, svm_y_train, svm_y_val = train_test_split(svm_feats, labels, test_size=0.1)
        #perform svm
        print("Performing SVM...")
        if CLASSIFIER == 'SVM':
            svm_model = svm.LinearSVC(C=C,verbose=1, max_iter=MAX_ITER)
            svm_model.fit(svm_x_train, svm_y_train)
        print (model)
        del svm_x_train
        del svm_y_train
        with open(MODEL_FILE, 'wb') as mf:
            pickle.dump(svm_model, mf)
        val_preds = svm_model.predict(svm_x_val)
        accuracy = metrics.accuracy_score(svm_y_val, val_preds)
        print("CNN-SVM Accuracy: %.2f%%" % (accuracy * 100.0))
        print("CNN-SVM Precision: %.2f%%" % (metrics.precision_score(svm_y_val, val_preds, average='weighted') * 100.0))
        print("CNN-SVM Recall: %.2f%%" % (metrics.recall_score(svm_y_val, val_preds, average='weighted') * 100.0))
        print("CNN-SVM F1-Score: %.2f%%" % (metrics.f1_score(svm_y_val, val_preds, average='weighted') * 100.0))
        print(metrics.classification_report(svm_y_val, val_preds, digits=4))
        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        model = load_model(sys.argv[1])
        print (model.summary())
        test_tweets, _ = process_tweets(TEST_PROCESSED_FILE, test_file=True)
        test_tweets = pad_sequences(test_tweets, maxlen=max_length, padding='post')
        print('Testing....')
        predictions = model.predict(test_tweets, batch_size=1024, verbose=1)
        with open(MODEL_FILE, 'rb') as mf:
            svm_model = pickle.load(mf)
        print (predictions.shape)
        test_preds = svm_model.predict(predictions)
        with open(TEST_PROCESSED_FILE, 'r', encoding="utf8") as csv:
            lines = csv.readlines()
            tweet_list = list(lines)
        results = zip(tweet_list, test_preds)
        utils.save_results_to_csv(results, 'cnn-feats-svm-linear-%.2f-%d.csv' % (C, MAX_ITER))
