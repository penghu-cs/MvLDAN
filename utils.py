import sys
import numpy as np
import theano
import theano.tensor as T
import time


def show_progressbar(rate, *args, **kwargs):
    '''
    :param rate: [current, total]
    :param args: other show
    '''
    inx = rate[0]
    count = rate[1]
    rate[0] = int(np.around(rate[0] * 50. / rate[1])) if rate[1] > 50 else rate[0]
    rate[1] = 50 if rate[1] > 50 else rate[1]
    num = len(str(count))
    str_show = ('\r%' + str(num) + 'd / ' + '%' + str(num) + 'd  (%' + '3.2f%%) [') % (inx, count, float(inx) / count * 100)
    for i in range(rate[0]):
        str_show += '='

    if rate[0] < rate[1] - 1:
        str_show += '>'

    for i in range(rate[0], rate[1] - 1, 1):
        str_show += '.'
    str_show += '] '
    for l in args:
        str_show += ' ' + str(l)

    for key in kwargs:
        try:
            str_show += ' ' + key + ': %.2f' % kwargs[key]
        except Exception:
            str_show += ' ' + key + ': ' + str(kwargs[key])

    sys.stdout.write(str_show)
    sys.stdout.flush()
#    time.sleep(0.2)


def th_knn(train, train_labels, test, batch=1000):
    batch_num = int(np.ceil(test.shape[0] / float(batch)))
    test_labels = []
    tr = theano.shared(train.astype('float32'))
    train_labels = train_labels.reshape([-1])
    for i in range(batch_num):
        te = theano.shared(test[i * batch: (i + 1) * batch].astype('float32'))
        # tr_la = theano.shared(train_labels.reshape([-1]).astype('int32'))
        cov = T.dot(te, tr.T) / T.dot(T.sqrt(T.sum(te ** 2, axis=1).reshape([-1, 1])), T.sqrt(T.sum(tr ** 2, axis=1).reshape([1, -1])))
        inx = T.argmax(cov, axis=1).reshape([-1])
        inx = theano.function([], inx)()
        test_labels.append(train_labels[inx])
    test_labels = np.concatenate(test_labels)
    # inx = T.argmax(cov, axis=1).reshape([-1])
    # test_labels = theano.function([], tr_la[inx])()

    return test_labels


def ZeroMeanOneVar(data):
    dtype = data.dtype;
    _mean = np.mean(data, axis=0).reshape([1, -1])
    _std = np.std(data, axis=0).reshape([1, -1])
    _std += np.equal(_std, 0).astype(dtype)
    return (data - _mean) / _std


def th_ZeroMeanOneVar(data):
    dtype = data.dtype;
    _mean = T.mean(data, axis=0).reshape([1, -1])
    _std = T.std(data, axis=0).reshape([1, -1])
    _std += T.eq(_std, 0).astype(dtype)
    return (data - _mean) / _std

def th_fx_calc_map_label2(train, train_labels, test, test_label, k=0):
    length = test_label.shape[0]
    # query train
    tr = theano.shared(train.astype('float32'))
    te = theano.shared(test.astype('float32'))
    tr_lab = theano.shared(train_labels.reshape([-1, 1]).astype('float32'))
    te_lab = theano.shared(test_label.reshape([-1, 1]).astype('float32'))
    # tr = th_ZeroMeanOneVar(tr)
    # te = th_ZeroMeanOneVar(te)
    # tr_la = theano.shared(train_labels.reshape([-1]).astype('int32'))
    dist = -T.dot(tr, te.T) / T.dot(T.sqrt(T.sum(tr ** 2, axis=1).reshape([-1, 1])),
                                    T.sqrt(T.sum(te ** 2, axis=1).reshape([1, -1])))

    if k == 0:
        k = length
    if k == -1:
        ks = [50, length]
    else:
        ks = [k]

    def calMAP(_k):
        inx = T.argsort(dist, axis=1)
        # A = (te_lab == tr_lab[inx[:, 0: _k].reshape([-1])].reshape([length, _k])).astype('float32')
        A = T.eq(te_lab, tr_lab[inx[:, 0: _k].reshape([-1])].reshape([length, _k])).astype('float32')
        U = T.triu(T.ones([_k, _k]))
        B = T.dot(A, U)
        B *= A
        r = T.sum(A, axis=1)
        p = T.sum(B / (T.arange(1, _k + 1).astype('float32')), axis=1)
        r, p = theano.function([], [r, p])()
        p = p[r.nonzero()]
        r = r[r.nonzero()]
        res = T.sum(p / r)
        res /= (_k * length)
        res = theano.function([], res)()
        return res

        # for i in range(numcases):
        #     order = ord[i]
        #     p = 0.0
        #     r = 0.0
        #     for j in range(_k):
        #         if test_label[i] == train_labels[order[j]]:
        #             r += 1
        #             p += (r / (j + 1))
        #     if r > 0:
        #         _res += [p / r]
        #     else:
        #         _res += [0]
        # return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))

    return res


def th_fx_calc_map_label(train, train_labels, test, test_label, k=0):

    # query train
    # tr = theano.shared(train.astype('float32'))
    # te = theano.shared(test.astype('float32'))
    # tr = th_ZeroMeanOneVar(tr)
    # te = th_ZeroMeanOneVar(te)
    # tr_la = theano.shared(train_labels.reshape([-1]).astype('int32'))

    # dist = -T.dot(tr, te.T) / T.dot(T.sqrt(T.sum(tr ** 2, axis=1).reshape([-1, 1])), T.sqrt(T.sum(te ** 2, axis=1).reshape([1, -1])))
    # dist = theano.function([], dist)()

    import scipy
    dist = scipy.spatial.distance.cdist(test, train, 'cosine')
    ord = np.argsort(dist, axis=1)
    
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(numcases):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))

    return res


def fx_calc_map_label(train, train_labels, test, test_label, k = 0, dist_method='L2'):
    import scipy
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(train, test, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(train, test, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if test_label[i] == train_labels[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def np_knn(train, train_labels, test):
    # tr_la = theano.shared(train_labels.reshape([-1]).astype('int32'))
    cov = np.matmul(test, train.T) / np.matmul(np.sqrt(np.sum(test ** 2, axis=1).reshape([-1, 1])), np.sqrt(np.sum(train ** 2, axis=1).reshape([1, -1])))
    inx = np.argmax(cov, axis=1).reshape([-1])
    test_labels = train_labels.reshape([-1])[inx]
    return test_labels

