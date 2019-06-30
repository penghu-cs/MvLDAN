from keras.callbacks import Callback
from data import load_wiki_raw_feature_10, load_nus_feature_10, load_voc_feature_10, load_nus_mnist_feature, load_raw_nus_mnist_feature
from model import create_nus_model, create_voc_model, create_mnist_cnn_model
from MvLDAN import th_MvLDAN_test, th_MvLDAN_test_w, th_MvLDAN_cost, th_MvLDAN
import config
import numpy as np
import scipy.io as sio


def train_model(model, data, epoch_num, batch_size, out_model=None, pairwise=True, d=9, MAP=None, model_path='tmp/tmp_best2.h5'):
    str_test = [0]
    best_val_accuracy = [0]
    best_test_accuracy = [0]
    result = []
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []
    isComputeLoss = False
    # MAP = MAP # None
    compute_all = False
    tmp_best = model_path
    best_epoch = [0]
    for i in range(len(data)):
        train_data.append(data[i][0][0])
        train_labels.append(np.reshape(data[i][0][1], [-1, 1]))
        valid_data.append(data[i][1][0])
        valid_labels.append(np.reshape(data[i][1][1], [-1, 1]))
        test_data.append(data[i][2][0])
        test_labels.append(np.reshape(data[i][2][1], [-1, 1]))

    class LossHistory(Callback):
        def __init__(self, _train, _validation, _test, _batch_size=100, d=9):
            self.train_data = _train[0]
            self.train_labels = _train[1]

            self.validate_data = _validation[0]
            self.validate_labels = _validation[1]

            self.test_data = _test[0]
            self.test_labels = _test[1]

            self.batch_size = _batch_size
            self.n_view = len(self.train_data)
            self.d = d

            if out_model is None:
                self.out_model = self.model
            else:
                self.out_model = out_model

            self.history = {'tr_eigvals': [], 'val_eigvals': [], 'tr_acc': [], 'val_acc': []}

            self.test_pred = None
            self.train_pred = None

        def on_train_begin(self, logs={}):
            if isComputeLoss:
                _train = self.out_model.predict(self.train_data, self.batch_size)
                _validate = self.out_model.predict(self.validate_data, self.batch_size)
                _val_result, tr_eigvals, _, W, ms = th_MvLDAN_test(_train, self.train_labels, _validate, self.validate_labels, self.d, MAP)

                _train_resut = th_MvLDAN_test_w(W, ms, _train, self.train_labels, self.d, MAP)
                _, val_eigvals, _, _, _ = th_MvLDAN_test(_validate, self.validate_labels, _validate, self.validate_labels, self.d, MAP)
                self.history['tr_eigvals'].append(tr_eigvals)
                self.history['val_eigvals'].append(val_eigvals)
                self.history['tr_acc'].append(_train_resut)
                self.history['val_acc'].append(_val_result)
            pass
            self.on_epoch_end(-1)

        def on_batch_end(self, batch, logs={}):
            pass

        def view_result(self, _acc):
            res = ''
            if type(_acc) is not list:
                res += ((' - mean: %.4f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
                for _i in range(self.n_view):
                    for _j in range(self.n_view):
                        if _i != _j:
                            res += ('%.4f' % _acc[_i, _j]) + ','
            else:
                R = [50, 'ALL']
                for _k in range(len(_acc)):
                    res += (' R = ' + str(R[_k]) + ': ')
                    res += ((' - mean: %.4f' % (np.sum(_acc[_k]) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
                    for _i in range(self.n_view):
                        for _j in range(self.n_view):
                            if _i != _j:
                                res += ('%.4f' % _acc[_k][_i, _j]) + ','
            return res

        def on_epoch_end(self, epoch, logs=None):
            _train = self.out_model.predict(self.train_data, self.batch_size)
            _validate = self.out_model.predict(self.validate_data, self.batch_size)
            _val_result, tr_eigvals, _, W, ms = th_MvLDAN_test(_train, self.train_labels, _validate, self.validate_labels, self.d, MAP)#list(range(2, 30)))

            val_eigvals_sum = np.sum(tr_eigvals[0::])
            self.str_test = ''
            if compute_all or np.sum(_val_result) > np.sum(best_val_accuracy[0]):
                best_val_accuracy[0] = _val_result
                self.train_pred = _train
                self.test_pred = self.out_model.predict(self.test_data, self.batch_size)
            print(' - val_sum: %.4f - val_results: %s %s  - val_eigenvalues: %.4f %.4f' % (val_eigvals_sum, self.view_result(_val_result), self.str_test, tr_eigvals[0], tr_eigvals[-1]))
            _val_tmp = np.concatenate(_val_result)
            result.append(np.sum(_val_result) / len(_val_tmp[_val_tmp.nonzero()]))

            if isComputeLoss:
                _train_resut = th_MvLDAN_test_w(W, ms, _train, self.train_labels, self.d, MAP)
                _, val_eigvals, _, _, _ = th_MvLDAN_test(_validate, self.validate_labels, _validate, self.validate_labels, self.d, MAP=MAP)
                self.history['tr_eigvals'].append(tr_eigvals)
                self.history['val_eigvals'].append(val_eigvals)
                self.history['tr_acc'].append(_train_resut)
                self.history['val_acc'].append(_val_result)
    print('start training...........')
    if pairwise is True:
        history = LossHistory([train_data, train_labels], [valid_data, valid_labels], [test_data, test_labels], _batch_size=batch_size, d=d)
        H = model.fit(train_data + train_labels, train_labels[0], batch_size=batch_size, epochs=epoch_num, shuffle=True, callbacks=[history], verbose=1)
        if isComputeLoss:
            import scipy.io as sio
            history.history['tr_loss'] = H.history['loss']
            sio.savemat('cnn_loss_acc_history_noisy_mnist_20.mat', history.history)
            exit(0)
    else:
        from model import batch_generator
        model.fit_generator(batch_generator(data), steps_per_epoch=batch_size, epochs=epoch_num, validation_data=batch_generator(data, 1), validation_steps=batch_size, callbacks=[LossHistory([train_data, train_labels], [valid_data, valid_labels], [test_data, test_labels], _batch_size=batch_size, d=d)])

    tr = history.train_pred
    te = history.test_pred

    import os
    import scipy.io as sio
    for i in range(1, 100):
        file_name = config.feature_path + '_' + str(i) + '.mat'
        if not os.path.exists(file_name):
            ms, W, eigvals = th_MvLDAN(tr, train_labels)
            test_list = []
            for v in range(len(train_labels)):
                test_list.append(np.dot((te[v] - ms[0][v]) / ms[1][v], W[v][:, 0:d]))
                # test_list.append(np.dot(te[v], W[v][:, 0:d]))
            if len(train_labels) == 2:
                sio.savemat(file_name, {'img': test_list[0], 'txt': test_list[1], 'img_lab': test_labels[0], 'txt_lab': test_labels[1]})
            else:
                sio.savemat(file_name, {'test': np.array(test_list), 'labels': np.array(test_labels)})
            break
    print('best_epoch:' + str(best_epoch[0]) + 'max mean accuracy:' + str(np.max(result)) + str(str_test[0]))
    return {'valid_max': best_val_accuracy[0], 'test_result': best_test_accuracy[0]}


def pretrain(model, data, epoch_num, batch_size, out_model=None, pairwise=True, d=9):
    str_test = [0]
    best_val_accuracy = [0]
    best_test_accuracy = [0]
    result = []
    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []
    isComputeLoss = False
    for i in range(len(data)):
        train_data.append(data[i][0][0])
        train_labels.append(np.reshape(data[i][0][1], [-1, 1]))
        valid_data.append(data[i][1][0])
        valid_labels.append(np.reshape(data[i][1][1], [-1, 1]))
        test_data.append(data[i][2][0])
        test_labels.append(np.reshape(data[i][2][1], [-1, 1]))
    print('start pretraining...........')
    model.fit(train_data + train_labels, train_labels[0], batch_size=batch_size, epochs=epoch_num, shuffle=True)

def train_nus(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    all_data = load_nus_feature_10()
    result = []
    if type(all_data) is tuple:
        all_data = [all_data]
        input_size = all_data[0][1]
    else:
        _input_size = all_data[0][1]
        input_size = []
        for i in _input_size:
            input_size.append(tuple(i.reshape([-1]).tolist()))
    times = 1
    if config.test_times == -1:
        times = len(all_data)
    for index in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = index
        else:
            inx = config.test_times
        dd = all_data[inx]
    # for dd in all_data:
        _all_data = dd[0]
        model, predit_model = create_nus_model(input_size, output_size, l2, learning_rate)
        model.summary()
        print("lambda_cca1: " + str(config.lambda_cca1) + '            index: ' + str(inx))
        result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, MAP=-1, d=d, model_path='tmp/nus_model.h5'))
    return result


def train_voc(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=19):
    # from data import load_voc_ccl_feature
    all_data = load_voc_feature_10()
    # all_data = load_voc_ccl_feature()
    result = []
    if type(all_data) is tuple:
        all_data = [all_data]
        input_size = all_data[0][1]
    else:
        _input_size = all_data[0][1]
        input_size = []
        for i in _input_size:
            input_size.append(tuple(i.reshape([-1]).tolist()))
    times = 1
    if config.test_times == -1:
        times = len(all_data)
    for index in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = index
        else:
            inx = config.test_times
        dd = all_data[inx]
        _all_data = dd[0]
        model, predit_model = create_voc_model(input_size, output_size, l2, learning_rate)
        model.summary()
        print("lambda_cca1: " + str(config.lambda_cca1) + '       index: ' + str(inx))
        result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, MAP=-1, model_path='tmp/voc_model.h5'))
    return result

def train_mnist_cnn(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    all_inx = sio.loadmat('./data/mnist/mnist_shuffle_inx10.mat')['mnist_shuffle_inx10']
    result = []
    times = 1
    if config.test_times == -1:
        times = len(all_inx)
    for i in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = i
        else:
            inx = config.test_times
        from data import load_mnist
        all_data = load_mnist(all_inx[inx, :], D=2)
        if type(all_data) is tuple:
            all_data = [all_data]
            input_size = all_data[0][1]
        else:
            _input_size = all_data[0][1]
            input_size = []
            for _i in _input_size:
                input_size.append(tuple(_i.reshape([-1]).tolist()))

        dd = all_data[0]
        _all_data = dd[0]
        model, predit_model = create_mnist_cnn_model(input_size, output_size, l2, learning_rate)
        model.summary()
        # print("lambda_cca1: " + str(config.lambda_cca1))
        print("lambda_cca1: " + str(config.lambda_cca1) + '       index: ' + str(inx))
        result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, model_path='tmp/mnist_cnn_model.h5'))
    return result


def train_mnist_cnn_lambda(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    all_inx = sio.loadmat('./data/mnist/mnist_shuffle_inx10.mat')['mnist_shuffle_inx10']
    result = []
    times = 1
    if config.test_times == -1:
        times = len(all_inx)
    for i in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = i
        else:
            inx = config.test_times
        from data import load_mnist
        all_data = load_mnist(all_inx[inx, :], D=2)
        if type(all_data) is tuple:
            all_data = [all_data]
            input_size = all_data[0][1]
        else:
            _input_size = all_data[0][1]
            input_size = []
            for _i in _input_size:
                input_size.append(tuple(_i.reshape([-1]).tolist()))
        for dd in all_data:
            _all_data = dd[0]
            model, predit_model = create_mnist_cnn_model(input_size, output_size, l2, learning_rate)
            model.summary()
            print("lambda_cca1: " + str(config.lambda_cca1) + '       index: ' + str(inx))
            result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, model_path='tmp/mnist_cnn_lambda_model.h5'))
    return result


def train_mnist_full(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    all_inx = sio.loadmat('./data/mnist/mnist_shuffle_inx10.mat')['mnist_shuffle_inx10']
    result = []
    times = 1
    if config.test_times == -1:
        times = len(all_inx)
    for i in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = i
        else:
            inx = config.test_times
        from data import load_mnist
        all_data = load_mnist(all_inx[inx, :], D=1)

        if type(all_data) is tuple:
            all_data = [all_data]
            input_size = all_data[0][1]
        else:
            _input_size = all_data[0][1]
            input_size = []
            for _i in _input_size:
                input_size.append(tuple(_i.reshape([-1]).tolist()))
        for dd in all_data:
            _all_data = dd[0]
            from model import create_mnist_full_model
            model, predit_model = create_mnist_full_model(input_size, output_size, l2, learning_rate)
            model.summary()
            print("lambda_cca1: " + str(config.lambda_cca1) + '       index: ' + str(inx))
            result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, model_path='tmp/mnist_full_model.h5'))
    return result


def train_noisy_mnist_full(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    result = []
    for i in range(1):
        from data import load_noisyMNIST
        all_data = load_noisyMNIST()

        if type(all_data) is tuple:
            all_data = [all_data]
            input_size = all_data[0][1]
        else:
            _input_size = all_data[0][1]
            input_size = []
            for i in _input_size:
                input_size.append(tuple(i.reshape([-1]).tolist()))
        for dd in all_data:
            _all_data = dd[0]
            from model import create_mnist_full_model
            model, predit_model = create_mnist_full_model(input_size, output_size, l2, learning_rate)
            model.summary()
            print("lambda_cca1: " + str(config.lambda_cca1))
            result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, model_path='tmp/noisy_mnist_full_model.h5'))
    return result


def train_noisy_mnist_cnn(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    result = []
    for i in range(1):
        from data import load_noisyMNIST
        all_data = load_noisyMNIST(D=2)
        if type(all_data) is tuple:
            all_data = [all_data]
            input_size = all_data[0][1]
        else:
            _input_size = all_data[0][1]
            input_size = []
            for i in _input_size:
                input_size.append(tuple(i.reshape([-1]).tolist()))
        for dd in all_data:
            _all_data = dd[0]
            model, predit_model = create_mnist_cnn_model(input_size, output_size, l2, learning_rate)
            model.summary()
            print("lambda_cca1: " + str(config.lambda_cca1))
            result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, model_path='tmp/noisy_mnist_cnn_model.h5'))
    return result

def train_nMSAD_CNN(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=9):
    # all_inx = sio.loadmat('./data/mnist_cifar10/mnist_cifar10_shuffle_inx10.mat')['mnist_cifar10_shuffle_inx10']
    result = []
    from model import create_nMSAD_CNN_model
    times = 1
    if config.test_times == -1:
        times = 10
    for i in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = i
        else:
            inx = config.test_times
    # for i in range(10):
        from data import load_nMSAD
        all_data = load_nMSAD(inx, D=2)
        if type(all_data) is tuple:
            all_data = [all_data]
            input_size = all_data[0][1]
        else:
            _input_size = all_data[0][1]
            input_size = []
            for _i in _input_size:
                input_size.append(tuple(_i.reshape([-1]).tolist()))
        # for index in range(len(all_data)):
        # for dd in all_data:
        dd = all_data[0]
        _all_data = dd[0]
        model, predit_model = create_nMSAD_CNN_model(input_size, output_size, l2, learning_rate)
        model.summary()
        print("lambda_cca1: " + str(config.lambda_cca1) + '       index: ' + str(inx))
        result.append(train_model(model, _all_data, epoch_num, batch_size, predit_model, d=d, model_path='tmp/MNIST_spoken_cnn_model.h5'))
    return result