from keras.models import Sequential, Model, Input
from keras.layers import Dense, Lambda, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam, SGD
from MvLDAN import MvLDAN_gneral

def create_nus_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)

    models.append(Sequential())
    models[0].add(Dense(256, input_shape=input_size[0], activation='relu', kernel_regularizer=l2(value_l2)))
    models[0].add(Dense(256, activation='relu', kernel_regularizer=l2(value_l2)))
    models[0].add(Dense(128, activation='relu', kernel_regularizer=l2(value_l2)))
    models[0].add(Dense(output_size, kernel_regularizer=l2(value_l2)))

    models.append(Sequential())
    models[1].add(Dense(256, input_shape=input_size[1], activation='relu', kernel_regularizer=l2(value_l2)))
    models[1].add(Dense(256, activation='relu', kernel_regularizer=l2(value_l2)))
    models[1].add(Dense(128, activation='relu', kernel_regularizer=l2(value_l2)))
    models[1].add(Dense(output_size, kernel_regularizer=l2(value_l2)))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)


def create_voc_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)

    models.append(Sequential())
    models[0].add(Dense(2048, input_shape=input_size[0], activation='relu', kernel_regularizer=l2(value_l2)))
    #models[0].add(Dense(1024, activation='relu', kernel_regularizer=l2(value_l2)))
    #models[0].add(Dense(1024, activation='relu', kernel_regularizer=l2(value_l2)))
    models[0].add(Dense(output_size, kernel_regularizer=l2(value_l2)))

    models.append(Sequential())
    models[1].add(Dense(2048, input_shape=input_size[1], activation='relu', kernel_regularizer=l2(value_l2)))
    #models[1].add(Dense(1024, activation='relu', kernel_regularizer=l2(value_l2)))
    #models[1].add(Dense(1024, activation='relu', kernel_regularizer=l2(value_l2)))
    models[1].add(Dense(output_size, kernel_regularizer=l2(value_l2)))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)

def create_mnist_cnn_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)


    # view1 - mnist-view1
    models.append(Sequential())
    models[0].add(Conv2D(32, (3, 3), input_shape=input_size[0], padding='same'))
    # models[0].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(Conv2D(64, (3, 3), padding='same'))
    # models[0].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(MaxPool2D((2, 2), padding='same'))

    models[0].add(Conv2D(64, (3, 3), padding='same'))
    # models[0].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(Conv2D(64, (3, 3), padding='same'))
    # models[0].add(BatchNormalization())
    # models[0].add(Activation('relu'))
    models[0].add(MaxPool2D((2, 2), padding='same'))

    # Fully connected layer
    models[0].add(Flatten())
    models[0].add(Dense(128))
    # models[0].add(BatchNormalization())
    models[0].add(Activation('relu'))
    # models[0].add(Dropout(0.2))
    models[0].add(Dense(output_size))

    # view2 - mnist-view2
    models.append(Sequential())
    models[1].add(Conv2D(32, (3, 3), input_shape=input_size[1], padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(MaxPool2D((2, 2), padding='same'))

    models[1].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(MaxPool2D((2, 2), padding='same'))

    # Fully connected layer
    models[1].add(Flatten())
    models[1].add(Dense(128))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    # models[1].add(Dropout(0.2))
    models[1].add(Dense(output_size))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)


def create_mnist_full_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)

    models.append(Sequential())
    models[0].add(Dense(256, input_shape=input_size[0], activation='relu'))
    models[0].add(Dense(256, activation='relu'))
    models[0].add(Dense(128, activation='relu'))
    models[0].add(Dense(output_size))

    models.append(Sequential())
    models[1].add(Dense(256, input_shape=input_size[1], activation='relu'))
    models[1].add(Dense(256, activation='relu'))
    models[1].add(Dense(128, activation='relu'))
    models[1].add(Dense(output_size))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)

def create_nMSAD_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)
    # view1 - nus_imgs
    models.append(Sequential())
    models[0].add(Dense(512, input_shape=input_size[0], activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    # models[0].add(Dense(512, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    models[0].add(Dense(128, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    models[0].add(Dense(output_size, kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))

    # view2 - nus_tags
    models.append(Sequential())
    models[1].add(Dense(512, input_shape=input_size[1], activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    # models[1].add(Dense(512, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    models[1].add(Dense(128, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    models[1].add(Dense(output_size, kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))

    # view3 - mnist-view1
    models.append(Sequential())
    models[2].add(Dense(512, input_shape=input_size[2], activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    # models[2].add(Dense(512, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    models[2].add(Dense(128, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
    models[2].add(Dense(output_size, kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)


def create_nMSAD_CNN_model(input_size, output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = len(input_size)

    # mnist
    # view1
    from keras.layers import LSTM
    from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
    models.append(Sequential())

    models[0].add(Conv2D(32, (3, 3), input_shape=input_size[0], padding='same'))
    # models[1].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(MaxPool2D((2, 2), padding='same'))

    models[0].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[0].add(Activation('relu'))
    models[0].add(MaxPool2D((2, 2), padding='same'))

    # Fully connected layer
    models[0].add(Flatten())
    models[0].add(Dense(128))
    # models[0].add(BatchNormalization())
    models[0].add(Activation('relu'))
    # models[0].add(Dropout(0.2))
    models[0].add(Dense(output_size))

    # view2
    models.append(Sequential())
    models[1].add(Conv2D(32, (3, 3), input_shape=input_size[1], padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(MaxPool2D((2, 2), padding='same'))

    models[1].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    models[1].add(MaxPool2D((2, 2), padding='same'))

    # Fully connected layer
    models[1].add(Flatten())
    models[1].add(Dense(128))
    # models[1].add(BatchNormalization())
    models[1].add(Activation('relu'))
    # models[1].add(Dropout(0.2))
    models[1].add(Dense(output_size))



    # cifar
    # view3
    models.append(Sequential())
    models[2].add(Conv2D(32, (3, 3), input_shape=input_size[2], padding='same'))
    # models[0].add(BatchNormalization())
    models[2].add(Activation('relu'))
    models[2].add(Conv2D(64, (3, 3), padding='same'))
    # models[0].add(BatchNormalization())
    models[2].add(Activation('relu'))
    models[2].add(MaxPool2D((2, 2), padding='same'))

    models[2].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[2].add(Activation('relu'))
    models[2].add(Conv2D(64, (3, 3), padding='same'))
    # models[1].add(BatchNormalization())
    models[2].add(Activation('relu'))
    models[2].add(MaxPool2D((2, 2), padding='same'))


    # Fully connected layer
    models[2].add(Flatten())
    models[2].add(Dense(128))
    # models[0].add(BatchNormalization())
    models[2].add(Activation('relu'))
    # models[0].add(Dropout(0.2))
    models[2].add(Dense(output_size))


    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)
