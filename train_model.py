"""
    Name:           train_model.py
    Created:        11/7/2017
    Description:    Train LSTM on instacart.
"""
#==============================================
#                   Modules
#==============================================
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[1]
import pandas as pd
import numpy as np
import gzip
import pickle
import json
from keras.preprocessing.image import Iterator
from keras.models import Model, model_from_json, Sequential
from keras.layers import Dense, LSTM
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
#==============================================
#                   Files
#==============================================


#==============================================
#                   Classes
#==============================================
class InstacartIterator(Iterator):
    """Iterator yielding data for instacart
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
    """

    def __init__(self, orders_df, users_dict, max_orders, n_products,
                 batch_size=32, shuffle=False, test_time=False, seed=None):

        self.orders_df = orders_df
        self.users_dict = users_dict

        self.max_orders = max_orders
        self.n_products = n_products

        self.test_time = test_time

        super(InstacartIterator, self).__init__(len(self.orders_df.index), batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        # The transformation of data is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size, self.max_orders-1, self.n_products+8), dtype=K.floatx())
        if not self.test_time:
            batch_y = np.zeros((current_batch_size, self.n_products), dtype=K.floatx())

        for i, j in enumerate(index_array):

            current_order = self.orders_df.iloc[j]

            current_history = self.users_dict[current_order.user_id]
            the_order = current_history[current_history.order_number == current_order.order_number]
            current_metadata = the_order[["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]].values
            current_metadata = np.repeat(current_metadata, current_order.order_number - 1, axis=0)
            past_orders = current_history[current_history.order_number < current_order.order_number]
            past_metadata = past_orders[["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]].values

            past_data = np.zeros((self.max_orders-1, self.n_products+8))
            for ix, on, pid, pw in past_orders[["order_number", "product_id", "add_to_cart_order"]].itertuples():
                row_coord = self.max_orders - 1 - (current_order.order_number - on)
                col_coords = [int(cc)-1 for cc in pid]
                past_data[row_coord, col_coords] = pw

            past_data[self.max_orders-current_order.order_number:,-8:-4] = past_metadata
            past_data[self.max_orders-current_order.order_number:,-4:] = current_metadata

            current_data = np.zeros((self.n_products,))
            current_data[[int(cc)-1 for cc in the_order.product_id[0]]] = 1.

            batch_x[i] = past_data
            batch_y[i] = current_data


        if self.test_time:
            return batch_x
        else:
            return batch_x, batch_y


#==============================================
#                   Functions
#==============================================
def import_and_process_data(dump=False, verbose=1):
    """
    Import data and process it.
    """

    ### read dfs
    df_aisles = pd.read_csv("../data/instacart/aisles.csv")
    df_departments = pd.read_csv("../data/instacart/departments.csv")
    df_prior = pd.read_csv("../data/instacart/order_products__prior.csv")
    df_train = pd.read_csv("../data/instacart/order_products__train.csv")
    df_orders = pd.read_csv("../data/instacart/orders.csv")
    df_products = pd.read_csv("../data/instacart/products.csv")
    df_sample_submission = pd.read_csv("../data/instacart/sample_submission.csv")

    ### change data type for ids
    for df in (df_aisles, df_departments, df_prior, df_train, df_orders, df_products):
        df[[cname for cname in df.columns if cname.endswith("_id")]] = \
        df[[cname for cname in df.columns if cname.endswith("_id")]].astype(str)

    ### encode products

    # train
    df_train_products = df_train.groupby("order_id").agg({"product_id": (lambda x: list(x)), "add_to_cart_order": (lambda x: list(x))})
    def reverse_order(l):
        l_max = max(l)+10.
        return [(1. - (ll)/l_max) for ll in l]
    df_train_products["add_to_cart_order"] = df_train_products["add_to_cart_order"].apply(reverse_order)

    # prior
    df_prior_products = df_prior.groupby("order_id").agg({"product_id": (lambda x: list(x)), "add_to_cart_order": (lambda x: list(x))})
    def reverse_order(l):
        l_max = max(l)+10.
        return [(1. - (ll)/l_max) for ll in l]
    df_prior_products["add_to_cart_order"] = df_prior_products["add_to_cart_order"].apply(reverse_order)

    # concat
    df_prior_train_products = pd.concat([df_train_products, df_prior_products])

    ### merge with other useful info
    df_orders = df_orders.set_index("order_id")
    df_prior_train_orders = df_prior_train_products.join(df_orders, how='left')

    ### to dict of users
    df_prior_train_users = df_prior_train_orders.reset_index()
    df_prior_train_users = df_prior_train_users.set_index("user_id")
    df_grouped_users = df_prior_train_users.groupby(by=(lambda x: x))
    users_dict = {}
    for group, df in df_grouped_users:
        users_dict[group] = df.set_index("order_id").sort_values("order_number").fillna(-1)

    ### get order list for training
    orders_df = df_prior_train_orders[df_prior_train_orders.order_number >= 4][["user_id", "order_number"]]

    ### number of products and orders
    max_orders = orders_df.order_number.max()
    n_products = df_products.product_id.shape[0]

    if dump:
        orders_df.to_csv("../data/instacart/orders_df.csv", index=True)
        with gzip.open("../data/instacart/users_dict.gzip", "wb") as iOF:
            pickle.dump(users_dict, iOF)

    return orders_df, users_dict




def fbs(y_true, y_pred, threshold_shift=0., beta=1):

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())





def train_lstm(users_dict, orders_df_train, orders_df_val,
               max_orders, n_products,
               n_units_lstm=32, lr=0.001,
               batch_size=32, epochs=1000, patience=10, patience_lr=3,
               weights_path="../data/instacart/models/lstm_keras_default.h5",
               model_path="../data/instacart/models/lstm_keras_default.json",
               verbose=1):
    """
    Instantiate lstm net and train it.
    """

    ##### Instantiate model and compile it
    data_dim = n_products + 8
    timesteps = max_orders - 1
    n_classes = n_products

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(n_units_lstm, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(n_units_lstm, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(n_units_lstm))  # return a single vector of dimension 32
    model.add(Dense(n_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=[fbs])

    # serialize model to json
    model_json = model.to_json()
    with open(model_path, "w") as iOF:
        iOF.write(model_json)

    ##### Make data generators
    train_generator = InstacartIterator(orders_df_train, users_dict, max_orders, n_products,
                                        batch_size=batch_size, shuffle=True)
    val_generator = InstacartIterator(orders_df_val, users_dict, max_orders, n_products,
                                      batch_size=batch_size, shuffle=True)

    ##### Feed data
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(orders_df_train.index) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(orders_df_val.index) // batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=patience),
                   ModelCheckpoint(filepath=weights_path, save_best_only=True),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=patience_lr)]
        )



def main(verbose=1):
    """
    Main function.
    """

    if verbose >= 1:
        if sys.argv[1] == "":
            print("========== TF is running on CPU ==========")
        else:
            print("========== TF is running on GPU %d =========="%(int(sys.argv[1])))

    ##### Imports
    if verbose >= 1: print("Importing data...")
    #orders_df, users_dict = import_and_process_data()
    orders_df = pd.read_csv("../data/instacart/orders_df.csv", index_col=0)
    orders_df.index = orders_df.index.astype(str)
    orders_df.user_id = orders_df.user_id.astype(str)
    with gzip.open("../data/instacart/users_dict.gzip", "rb") as iOF:
        users_dict = pickle.load(iOF)

    ##### Get parameters
    if verbose >= 1: print("Getting parameters...")
    df_products = pd.read_csv("../data/instacart/products.csv")
    n_products = df_products.product_id.shape[0]
    max_orders = orders_df.order_number.max()

    ##### Split train and val
    if verbose >= 1: print("Splitting train and val...")
    train_frac = .8
    orders_df = orders_df.sample(frac=1)
    split_point = int(train_frac*len(orders_df.index))
    orders_df_train = orders_df.iloc[:split_point]
    orders_df_val = orders_df.iloc[split_point:]

    #### Train
    if verbose >= 1: print("Training LSTM net...")
    train_lstm(users_dict, orders_df_train, orders_df_val,
               max_orders, n_products,
               n_units_lstm=32, lr=0.001,
               batch_size=512, epochs=1000, patience=10, patience_lr=3,
               weights_path="../data/instacart/models/lstm_keras_001.h5",
               model_path="../data/instacart/models/lstm_keras_001.json",
               verbose=verbose)



#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    main(1)
