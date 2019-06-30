import gzip
import numpy as np
import scipy.io as sio
import h5py
import pickle
import os.path

def store_data(path, data):
    n_bytes = 2**31
    max_bytes = 2**31 - 1
    # data = bytearray(n_bytes)

    ## write
    bytes_out = pickle.dumps(data)
    with open(path, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])
    # assert(data == data2)


def read_data(path):
    ## read
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(path)
    with open(path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data


def load_mnist():
    # mnist dataset
    INPUT_SHAPE = [28, 28]

    data1 = load_data('datasets/noisymnist_view1.gz')
    data2 = load_data('datasets/noisymnist_view2.gz')
    ret_input_size = []

    input_size = [INPUT_SHAPE[0], INPUT_SHAPE[1], 1]
    for i in range(len(data1)):
        data1[i][0] = data1[i][0].reshape([-1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1]).astype(np.float32)
        data2[i][0] = data2[i][0].reshape([-1, INPUT_SHAPE[0], INPUT_SHAPE[1], 1]).astype(np.float32)
    ret_input_size = [input_size, input_size]
    return [data1, data2], ret_input_size


def load_casia2(chunnel=1, train_enable=False, test_num=0):
    # casia 2.0 NIR VIS
    data = [[], []]
    INPUT_SHAPE = [128, 128]
    input_size = [INPUT_SHAPE[0], INPUT_SHAPE[1], chunnel]

    if train_enable is True:
        file_path = 'datasets/train_dev.mat'
    else:
        file_path = 'datasets/casia2_' + str(test_num) + '.mat'

    casia2 = sio.loadmat(file_path)['casia2'][0]
    vis_train_data_tmp = casia2[0][0][0][0].T.reshape([-1, INPUT_SHAPE[0], INPUT_SHAPE[1]]).astype(np.float32) / 256.
    vis_train_data = np.zeros(list(vis_train_data_tmp.shape) + [chunnel], dtype=np.float32)
    for i in range(chunnel):
        vis_train_data[:, :, :, i] = vis_train_data_tmp
    vis_train_labels = casia2[0][0][0][1].astype(np.int32)
    data[0].append([vis_train_data, vis_train_labels])

    nir_train_data_tmp = casia2[1][0][0][0].T.reshape([-1, INPUT_SHAPE[0], INPUT_SHAPE[1]]).astype(np.float32) / 256.
    # nir_mean = np.mean(nir_train_data_tmp, axis=0)
    # nir_std =np.std(nir_train_data_tmp, axis=0)
    # nir_train_data_tmp -= nir_mean  # zero-center
    # nir_train_data_tmp /= nir_std   # normalize
    nir_train_data = np.zeros(list(nir_train_data_tmp.shape) + [chunnel], dtype=np.float32)
    for i in range(chunnel):
        nir_train_data[:, :, :, i] = nir_train_data_tmp
    nir_train_labels = casia2[1][0][0][1].astype(np.int32)
    data[1].append([nir_train_data, nir_train_labels])

    vis_gallery_data_tmp = casia2[2][0][0][0].T.reshape([-1, INPUT_SHAPE[0], INPUT_SHAPE[1]]).astype(np.float32) / 256.
    # vis_gallery_data_tmp -= vis_mean  # zero-center
    # vis_gallery_data_tmp /= vis_std   # normalize
    vis_gallery_data = np.zeros(list(vis_gallery_data_tmp.shape) + [chunnel], dtype=np.float32)
    for i in range(chunnel):
        vis_gallery_data[:, :, :, i] = vis_gallery_data_tmp
    vis_gallery_labels = casia2[2][0][0][1].astype(np.int32)
    data[0].append([vis_gallery_data, vis_gallery_labels])

    nir_probe_data_tmp = casia2[3][0][0][0].T.reshape([-1, INPUT_SHAPE[0], INPUT_SHAPE[1]]).astype(np.float32) / 256.
    nir_probe_data = np.zeros(list(nir_probe_data_tmp.shape) + [chunnel], dtype=np.float32)
    for i in range(chunnel):
        nir_probe_data[:, :, :, i] = nir_probe_data_tmp
    nir_probe_labels = casia2[3][0][0][1].astype(np.int32)
    data[1].append([nir_probe_data, nir_probe_labels])

    ret_input_size = [input_size, input_size]
    return data, ret_input_size


def load_casia2_sift(train_enable=False, test_num=0):
    # casia 2.0 NIR VIS
    data = [[], []]
    # INPUT_SHAPE = [128 * 64, 128]

    if train_enable is True:
        file_path = 'datasets/train_sift_dev.mat'
    else:
        file_path = 'datasets/casia2_sift_' + str(test_num) + '.mat'

    casia2 = sio.loadmat(file_path)['casia2'][0]
    vis_train_data = casia2[0][0][0][0].T.astype(np.float32)
    input_size = [vis_train_data.shape[1],]
    vis_train_labels = casia2[0][0][0][1].astype(np.int32)
    data[0].append([vis_train_data, vis_train_labels])

    nir_train_data = casia2[1][0][0][0].T.astype(np.float32)
    nir_train_labels = casia2[1][0][0][1].astype(np.int32)
    data[1].append([nir_train_data, nir_train_labels])

    vis_gallery_data = casia2[2][0][0][0].T.astype(np.float32)
    vis_gallery_labels = casia2[2][0][0][1].astype(np.int32)
    data[0].append([vis_gallery_data, vis_gallery_labels])

    nir_probe_data = casia2[3][0][0][0].T.astype(np.float32)
    nir_probe_labels = casia2[3][0][0][1].astype(np.int32)
    data[1].append([nir_probe_data, nir_probe_labels])

    ret_input_size = [input_size, input_size]
    return data, ret_input_size


# def load_nus_wide():
#     nus = sio.loadmat('datasets/NUS-WIDE.mat')
#     train_imgs = nus['feax_tr'].astype('float32')
#     train_text = nus['feay_tr'].astype('float32')
#     train_labels = np.argmax(nus['gnd_tr'], axis=1)
#
#     test_imgs = nus['feax_te'].astype('float32')
#     test_text = nus['feay_te'].astype('float32')
#     test_labels = np.argmax(nus['gnd_te'], axis=1)
#
#     ret_input_size = [train_imgs.shape[1::], train_text.shape[1::]]
#     data = [[[train_imgs, train_labels], [test_imgs, test_labels]],
#                 [[train_text, train_labels], [test_text, test_labels]]]
#     return data, ret_input_size


def load_wiki_rbm_feature():
    import os
    data_dir = './data/wiki_data/rbm_reps_1024'
    train_data1 = np.load(os.path.join(data_dir, 'image_rbm2_LAST/train/image_hidden2-00001-of-00001.npy'))
    test_data1 = np.load(os.path.join(data_dir, 'image_rbm2_LAST/test/image_hidden2-00001-of-00001.npy'))
    tune_data1 = np.load(os.path.join(data_dir, 'image_rbm2_LAST/validation/image_hidden2-00001-of-00001.npy'))
    train_data2 = np.load(os.path.join(data_dir, 'text_rbm2_LAST/train/text_hidden2-00001-of-00001.npy'))
    test_data2 = np.load(os.path.join(data_dir, 'text_rbm2_LAST/test/text_hidden2-00001-of-00001.npy'))
    tune_data2 = np.load(os.path.join(data_dir, 'text_rbm2_LAST/validation/text_hidden2-00001-of-00001.npy'))
    train_labels = np.load(os.path.join(data_dir, 'label/train_lab_data.npy'))
    test_labels = np.load(os.path.join(data_dir, 'label/test_lab_data.npy'))
    tune_labels = np.load(os.path.join(data_dir, 'label/validation_lab_data.npy'))


    ret_input_size = [train_data1.shape[1::], train_data2.shape[1::]]
    data = [[[train_data1, train_labels], [tune_data1, tune_labels], [test_data1, test_labels]], [[train_data2, train_labels], [tune_data2, tune_labels], [test_data2, test_labels]]]
    return data, ret_input_size


def load_wiki_deep_feature():
    wiki = sio.loadmat('./data/wiki_deep_fea.mat')
    imgs = wiki['imgFea'].T
    texts = wiki['txtFea'].T
    labels = wiki['gnd'].reshape([-1])
    ret_input_size = [imgs.shape[1::], texts.shape[1::]]

    la = np.unique(labels)

    train_imgs_list = []
    train_imgs_labels_list = []

    train_texts_list = []
    train_texts_labels_list = []

    test_imgs_list = []
    test_imgs_labels_list = []

    test_texts_list = []
    test_texts_labels_list = []

    for i in range(la.shape[0]):
        inx = np.array(list(np.nonzero(labels == la[i]))).reshape([-1])
        np.random.shuffle(inx)
        train_inx = inx[0: 130]
        test_inx = inx[130::]

        train_imgs_list.append(imgs[train_inx, :])
        train_imgs_labels_list.append(labels[train_inx])

        train_texts_list.append(texts[train_inx, :])
        train_texts_labels_list.append(labels[train_inx])

        test_imgs_list.append(imgs[test_inx, :])
        test_imgs_labels_list.append(labels[test_inx])

        test_texts_list.append(texts[test_inx, :])
        test_texts_labels_list.append(labels[test_inx])

    train_imgs = np.concatenate(train_imgs_list, axis=0)
    mean_imgs = np.mean(train_imgs, axis=0).reshape([1, -1])
    var_imgs = np.var(train_imgs, axis=0).reshape([1, -1])
    # train_imgs = (train_imgs - mean_imgs) # / var_imgs
    train_imgs_labels = np.concatenate(train_imgs_labels_list, axis=0)

    train_texts = np.concatenate(train_texts_list, axis=0)
    mean_text = np.mean(train_texts, axis=0).reshape([1, -1])
    var_text = np.var(train_texts, axis=0).reshape([1, -1])
    # train_texts = (train_texts - mean_text) # / var_text
    train_texts_labels = np.concatenate(train_texts_labels_list, axis=0)

    test_imgs = np.concatenate(test_imgs_list, axis=0)
    # test_imgs = (test_imgs - mean_imgs) # / var_imgs
    test_imgs_labels = np.concatenate(test_imgs_labels_list, axis=0)

    test_texts = np.concatenate(test_texts_list, axis=0)
    # test_texts = (test_texts - mean_text) # / var_text
    test_texts_labels = np.concatenate(test_texts_labels_list, axis=0)

    data = [[[train_imgs, train_imgs_labels], [test_imgs, test_imgs_labels]],
                [[train_texts, train_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size


def load_wiki_raw_feature_10():
    wiki = sio.loadmat('./data/wiki10_rand.mat')
    return wiki['wiki10']

def load_wiki_features_text_pretrained():
    wiki = sio.loadmat('./data/wiki_data/wiki_features_text_pretrained.mat')
    train_imgs = wiki['X1']
    train_texts = wiki['X2']
    train_imgs_labels = wiki['trainLabel'].reshape([-1]).astype('uint8')
    train_texts_labels = wiki['trainLabel'].reshape([-1]).astype('uint8')

    validate_imgs = wiki['XV1']
    validate_texts = wiki['XV2']
    validate_imgs_labels = wiki['tuneLabel'].reshape([-1]).astype('uint8')
    validate_texts_labels = wiki['tuneLabel'].reshape([-1]).astype('uint8')

    test_imgs = wiki['XTe1']
    test_texts = wiki['XTe2']
    test_imgs_labels = wiki['testLabel'].reshape([-1]).astype('uint8')
    test_texts_labels = wiki['testLabel'].reshape([-1]).astype('uint8')
    ret_input_size = [train_imgs.shape[1::], train_texts.shape[1::]]
    data = [[[train_imgs, train_imgs_labels], [validate_imgs, validate_imgs_labels], [test_imgs, test_imgs_labels]],
                [[train_texts, train_texts_labels], [validate_texts, validate_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size
# load_wiki_features_text_pretrained()

def load_wiki_raw_feature():
    # wiki = sio.loadmat('datasets/wiki_feature.mat')
    # train_imgs = wiki['I_tr']
    # train_imgs_labels = wiki['I_tr_labels'].reshape([-1])
    #
    # train_texts = wiki['T_tr']
    # train_texts_labels = wiki['I_tr_labels'].reshape([-1])
    # test_imgs = wiki['I_te']
    # test_imgs_labels = wiki['I_te_labels'].reshape([-1])
    # test_texts = wiki['T_te']
    # test_texts_labels = wiki['T_te_labels'].reshape([-1])
    # ret_input_size = [[train_imgs.shape[1]], [train_texts.shape[1]]]
    # data = [[[train_imgs, train_imgs_labels], [test_imgs, test_imgs_labels]], [[train_texts, train_texts_labels], [test_texts, test_texts_labels]]]

    wiki = sio.loadmat('./data/wiki_feature_all.mat')
    imgs = wiki['imgs']
    texts = wiki['texts']
    labels = wiki['labels'].reshape([-1])
    ret_input_size = [imgs.shape[1::], texts.shape[1::]]

    la = np.unique(labels)

    train_imgs_list = []
    train_imgs_labels_list = []

    train_texts_list = []
    train_texts_labels_list = []

    test_imgs_list = []
    test_imgs_labels_list = []

    test_texts_list = []
    test_texts_labels_list = []

    validate_imgs_list = []
    validate_imgs_labels_list = []

    validate_texts_list = []
    validate_texts_labels_list = []
    # validate_num = 20
    for i in range(la.shape[0]):
        inx = np.array(list(np.nonzero(labels == la[i]))).reshape([-1])
        np.random.shuffle(inx)
        train_inx = inx[0: 130]
        # validate_inx = inx[130: 130 + validate_num]
        test_inx = inx[130::]

        train_imgs_list.append(imgs[train_inx, :])
        train_imgs_labels_list.append(labels[train_inx])

        train_texts_list.append(texts[train_inx, :])
        train_texts_labels_list.append(labels[train_inx])

        # validate_imgs_list.append(imgs[validate_inx, :])
        # validate_imgs_labels_list.append(labels[validate_inx])
        #
        # validate_texts_list.append(texts[validate_inx, :])
        # validate_texts_labels_list.append(labels[validate_inx])

        test_imgs_list.append(imgs[test_inx, :])
        test_imgs_labels_list.append(labels[test_inx])

        test_texts_list.append(texts[test_inx, :])
        test_texts_labels_list.append(labels[test_inx])

    train_imgs = np.concatenate(train_imgs_list, axis=0)
    train_imgs_labels = np.concatenate(train_imgs_labels_list, axis=0)

    train_texts = np.concatenate(train_texts_list, axis=0)
    train_texts_labels = np.concatenate(train_texts_labels_list, axis=0)

    # validate_imgs = np.concatenate(validate_imgs_list, axis=0)
    # validate_imgs_labels = np.concatenate(validate_imgs_labels_list, axis=0)
    #
    # validate_texts = np.concatenate(validate_texts_list, axis=0)
    # validate_texts_labels = np.concatenate(validate_texts_labels_list, axis=0)

    test_imgs = np.concatenate(test_imgs_list, axis=0)
    test_imgs_labels = np.concatenate(test_imgs_labels_list, axis=0)

    test_texts = np.concatenate(test_texts_list, axis=0)
    test_texts_labels = np.concatenate(test_texts_labels_list, axis=0)

    inx = np.array(range(test_imgs.shape[0]))
    np.random.shuffle(inx)
    test_imgs = test_imgs[inx]
    test_imgs_labels = test_imgs_labels[inx]
    test_texts = test_texts[inx]
    test_texts_labels = test_texts_labels[inx]

    validate_imgs = test_imgs[0: 200]
    validate_imgs_labels = test_imgs_labels[0: 200]

    validate_texts = test_texts[0: 200]
    validate_texts_labels = test_texts_labels[0: 200]

    test_imgs = test_imgs[200::]
    test_imgs_labels = test_imgs_labels[200::]
    test_texts = test_texts[200::]
    test_texts_labels = test_texts_labels[200::]

    data = [[[train_imgs, train_imgs_labels], [validate_imgs, validate_imgs_labels], [test_imgs, test_imgs_labels]],
                [[train_texts, train_texts_labels], [validate_texts, validate_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size


def load_nus_feature():
    val_num = 4655
    train_num = 1000

    nus = sio.loadmat('./data/nus_all.mat')
    imgs = nus['imgs'].astype('float32')
    texts = nus['texts'].astype('float32')
    labels = nus['labels'].reshape([-1])
    ret_input_size = [imgs.shape[1::], texts.shape[1::]]

    la = np.unique(labels)

    train_imgs_list = []
    train_imgs_labels_list = []

    train_texts_list = []
    train_texts_labels_list = []

    test_imgs_list = []
    test_imgs_labels_list = []

    test_texts_list = []
    test_texts_labels_list = []

    for i in range(la.shape[0]):
        inx = np.array(list(np.nonzero(labels == la[i]))).reshape([-1])
        np.random.shuffle(inx)
        train_inx = inx[0: train_num]
        test_inx = inx[train_num::]

        train_imgs_list.append(imgs[train_inx, :])
        train_imgs_labels_list.append(labels[train_inx])

        train_texts_list.append(texts[train_inx, :])
        train_texts_labels_list.append(labels[train_inx])

        test_imgs_list.append(imgs[test_inx, :])
        test_imgs_labels_list.append(labels[test_inx])

        test_texts_list.append(texts[test_inx, :])
        test_texts_labels_list.append(labels[test_inx])

    train_imgs = np.concatenate(train_imgs_list, axis=0)
    train_imgs_labels = np.concatenate(train_imgs_labels_list, axis=0)

    train_texts = np.concatenate(train_texts_list, axis=0)
    train_texts_labels = np.concatenate(train_texts_labels_list, axis=0)

    test_imgs = np.concatenate(test_imgs_list, axis=0)
    test_imgs_labels = np.concatenate(test_imgs_labels_list, axis=0)

    test_texts = np.concatenate(test_texts_list, axis=0)
    test_texts_labels = np.concatenate(test_texts_labels_list, axis=0)

    inx = np.array(range(test_imgs.shape[0]))
    np.random.shuffle(inx)
    test_imgs = test_imgs[inx]
    test_imgs_labels = test_imgs_labels[inx]
    test_texts = test_texts[inx]
    test_texts_labels = test_texts_labels[inx]

    validate_imgs = test_imgs[0: val_num]
    validate_imgs_labels = test_imgs_labels[0: val_num]

    validate_texts = test_texts[0: val_num]
    validate_texts_labels = test_texts_labels[0: val_num]

    test_imgs = test_imgs[val_num::]
    test_imgs_labels = test_imgs_labels[val_num::]
    test_texts = test_texts[val_num::]
    test_texts_labels = test_texts_labels[val_num::]

    data = [[[train_imgs, train_imgs_labels], [validate_imgs, validate_imgs_labels], [test_imgs, test_imgs_labels]],
                [[train_texts, train_texts_labels], [validate_texts, validate_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size


def load_nus_feature_10():
    nus10 = sio.loadmat('./data/nus10_20000.mat')
    return nus10['nus10']

def load_voc_ccl_feature():
    voc = sio.loadmat('./data/voc/Pascal_fea_whole.mat')

    train_imgs = voc['I_tr']
    train_imgs_labels = voc['I_tr_lab']

    train_texts = voc['T_tr']
    train_texts_labels = voc['T_tr_lab']

    validate_imgs = voc['I_va']
    validate_imgs_labels = voc['I_va_lab']

    validate_texts = voc['T_va']
    validate_texts_labels = voc['T_va_lab']

    test_imgs = voc['I_te']
    test_imgs_labels = voc['I_te_lab']

    test_texts = voc['T_te']
    test_texts_labels = voc['T_te_lab']

    ret_input_size = [train_imgs.shape[1::], train_texts.shape[1::]]
    data = [[[train_imgs, train_imgs_labels], [validate_imgs, validate_imgs_labels], [test_imgs, test_imgs_labels]],
            [[train_texts, train_texts_labels], [validate_texts, validate_texts_labels],
             [test_texts, test_texts_labels]]]
    return data, ret_input_size

def load_voc_feature():
    train_num = 3000
    val_num = 200

    nus = sio.loadmat('./data/pascal_voc.mat')
    imgs = nus['images'].astype('float32')
    texts = nus['texts'].astype('float32')
    labels = nus['labels'].reshape([-1])
    ret_input_size = [imgs.shape[1::], texts.shape[1::]]
    inx = np.array(range(imgs.shape[0]))
    np.random.shuffle(inx)
    imgs = imgs[inx, :]
    texts = texts[inx, :]
    labels = labels[inx]

    train_imgs = imgs[0: train_num, :]
    train_imgs_labels = labels[0: train_num]

    train_texts = texts[0: train_num, :]
    train_texts_labels = train_imgs_labels

    validate_imgs = imgs[train_num: train_num + val_num, :]
    validate_imgs_labels = labels[train_num: train_num + val_num]

    validate_texts = texts[train_num: train_num + val_num, :]
    validate_texts_labels = validate_imgs_labels

    test_imgs = imgs[train_num + val_num::, :]
    test_imgs_labels = labels[train_num + val_num::]

    test_texts = texts[train_num + val_num::, :]
    test_texts_labels = test_imgs_labels

    data = [[[train_imgs, train_imgs_labels], [validate_imgs, validate_imgs_labels], [test_imgs, test_imgs_labels]],
                [[train_texts, train_texts_labels], [validate_texts, validate_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size


def load_voc_feature_10():
    nus10 = sio.loadmat('./data/voc10.mat')
    return nus10['voc10']


def batch_generator(all_data, batch_size,  mode=0, train=False, shuffle=True):
    if train is True and mode == 1:
        mode = 2
    has_labels = False
    if isinstance(all_data[0], list):
        has_labels = True

    n_view = len(all_data)
    if mode == 0:
        if has_labels:
            length = all_data[0][0].shape[0]
        else:
            length = all_data[0].shape[0]
        vdata = []
        if shuffle is True:
            inx = np.array(list(range(length)))
            np.random.shuffle(inx)
            for i in range(n_view):
                # inx = np.array(list(range(length)))
                # np.random.shuffle(inx)
                # vdata.append([all_data[i][n_set][0][inx], all_data[i][n_set][1][inx]])
                if has_labels:
                    vdata.append([all_data[0][i][inx], np.reshape(np.reshape(all_data[1][i], [-1])[inx], [-1, 1])])
                else:
                    vdata.append(all_data[i][inx])
        else:
            for i in range(n_view):
                # vdata.append([all_data[i][n_set][0], all_data[i][n_set][1]])
                if has_labels:
                    vdata.append([all_data[0][i], np.reshape(all_data[1][i], [-1, 1])])
                else:
                    vdata.append(all_data[i])
        for i in range(0, length, batch_size):
            ret_data = []
            ret_labels = []
            for v in range(n_view):
                # ret_data.append(all_data[i][n_set][0][i: i + batch_size])
                # ret_labels.append(all_data[i][n_set][1][i: i + batch_size])
                if has_labels:
                    ret_data.append(vdata[v][0][i: i + batch_size])
                    ret_labels.append(vdata[v][1][i: i + batch_size])
                else:
                    ret_data.append(vdata[v][i: i + batch_size])
            if has_labels:
                yield ret_data, ret_labels
            else:
                yield ret_data
    elif mode == 1:
        la = []
        batch_list = []
        labels_list = []
        for i in range(n_view):
            # la.append(all_data[i][n_set][1])
            la.append(np.reshape(all_data[1][i], [-1]))
            batch_list.append([])
            labels_list.append([])
        la = np.concatenate(la, axis=0)
        la = np.unique(la)
        if shuffle is True:
            np.random.shuffle(la)
        inx = 0
        while inx < la.shape[0]:
            for i in range(n_view):
                batch_list[i].append(all_data[0][i][np.nonzero(all_data[1][i].reshape([-1]) == la[inx])].astype(np.float32))
                labels_list[i].append(all_data[1][i].reshape([-1, 1])[np.nonzero(all_data[1][i].reshape([-1]) == la[inx])].astype(np.int32))
            inx += 1
            if inx % batch_size == 0 or inx == la.shape[0]:
                ret_data = []
                ret_labels = []
                for i in range(n_view):
                    ret_data.append(np.concatenate(batch_list[i], axis=0))
                    ret_labels.append(np.concatenate(labels_list[i], axis=0).astype(np.int32))
                    batch_list[i] = []
                    labels_list[i] = []
                if has_labels:
                    yield ret_data, ret_labels
                else:
                    yield ret_data
    elif mode == 2:
        la = []
        batch_list = []
        labels_list = []
        for i in range(n_view):
            # la.append(all_data[i][n_set][1])
            la.append(np.reshape(all_data[1][i], [-1]))
            batch_list.append([])
            labels_list.append([])
        la = np.concatenate(la, axis=0)
        la = np.unique(la)
        if shuffle is True:
            np.random.shuffle(la)
        inx = 0

        la_num = np.zeros(la.shape, dtype=np.int32)
        indices = []
        for i in range(n_view):
            indices.append([])
            for inx in range(la.shape[0]):
                tmp = np.nonzero(all_data[1][i].reshape([-1]) == la[inx])
                indices[i].append(tmp)
                la_num[inx] = max(la_num[inx], tmp[0].shape[0])

        vdata = []
        length = int(np.sum(la_num))
        all_inx = np.array(list(range(length)))
        np.random.shuffle(all_inx)

        for i in range(n_view):
            for inx in range(la.shape[0]):
                ex_inx = indices[i][inx]
                if ex_inx[0].shape[0] < la_num[inx]:
                    add_inx = np.random.randint(0, ex_inx[0].shape[0], la_num[inx] - ex_inx[0].shape[0])
                    ex_inx = (np.concatenate([ex_inx[0], ex_inx[0][add_inx]]), )
                np.random.shuffle(ex_inx[0])
                batch_list[i].append(all_data[0][i][ex_inx].astype(np.float32))
                labels_list[i].append(all_data[1][i].reshape([-1, 1])[ex_inx].astype(np.int32))
            vdata.append([np.concatenate(batch_list[i])[all_inx], np.concatenate(labels_list[i])[all_inx]])

        for i in range(0, length, batch_size):
            ret_data = []
            ret_labels = []
            for v in range(n_view):
                ret_data.append(vdata[v][0][i: i + batch_size])
                ret_labels.append(vdata[v][1][i: i + batch_size])
            yield ret_data, ret_labels
    else:
        pass

def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle
    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)
    return ret


def make_numpy_array(data_xy):
    """converts the input to np arrays"""
    data_x, data_y = data_xy
    data_x = np.asarray(data_x, dtype='float32')
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def load_data(path):
    """loads the datasets from the gzip pickled files, and converts to np arrays"""
    print('loading datasets ...')
    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_numpy_array(train_set)
    valid_set_x, valid_set_y = make_numpy_array(valid_set)
    test_set_x, test_set_y = make_numpy_array(test_set)

    return [[train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]]


def dict2h5(filename, d):
    """Save a dictionary into an hdf5 file. Beware, the shape of arrays is reversed, 
       i.e. a (m,n,p) array is stored (and read back) as an (p,n,m) array.
       Complex arrays are stored as a dict with members 'r' and 'i' for real and 
       imaginary parts.
    """
    import h5py # must be installed
    if isinstance(filename, str):
        h = h5py.File(filename,'w') # create or truncate
    elif isinstance(filename, h5py.File) or isinstance(filename, h5py.Group):
        h = filename
    else:
        raise TypeError
    for k, v in d.items():
        if v is None:
            continue
        elif isinstance(v, list):
            h.create_dataset(k, data=np.array(v), compression="gzip")
        elif isinstance(v, (np.ndarray, np.int64, np.float64, str, bytes, np.float, float, np.float32,int)):
            try:
                h.create_dataset(k, data=v, compression="gzip")
            except TypeError:
                h.create_dataset(k, data=v)
        elif isinstance(v, dict):
            grp = h.create_group(k)
            dict2h5(v, filename=grp)


def load_nus_feature_2():
    train_num = 20000

    nus = sio.loadmat('./data/nus_all.mat')
    imgs = nus['imgs'].astype('float32')
    texts = nus['texts'].astype('float32')
    labels = nus['labels'].reshape([-1])
    ret_input_size = [imgs.shape[1::], texts.shape[1::]]
    inx = np.arange(imgs.shape[0])
    np.random.shuffle(inx)
    train_inx = inx[0: train_num]
    test_inx = inx[train_num::]

    train_imgs = imgs[train_inx, :]
    train_imgs_labels = labels[train_inx]
    train_texts = texts[train_inx, :]
    train_texts_labels = labels[train_inx]

    test_imgs = imgs[test_inx, :]
    test_imgs_labels = labels[test_inx]
    test_texts = texts[test_inx]
    test_texts_labels = labels[test_inx]

    data = [[[train_imgs, train_imgs_labels], [train_imgs, train_imgs_labels], [test_imgs, test_imgs_labels]], [[train_texts, train_texts_labels], [train_texts, train_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size


def load_voc_feature_2():
    train_num = 3000

    nus = sio.loadmat('./data/pascal_voc.mat')
    imgs = nus['images'].astype('float32')
    texts = nus['texts'].astype('float32')
    labels = nus['labels'].reshape([-1])
    ret_input_size = [imgs.shape[1::], texts.shape[1::]]
    inx = np.array(range(imgs.shape[0]))
    np.random.shuffle(inx)
    imgs = imgs[inx, :]
    texts = texts[inx, :]
    labels = labels[inx]

    train_imgs = imgs[0: train_num, :]
    train_imgs_labels = labels[0: train_num]

    train_texts = texts[0: train_num, :]
    train_texts_labels = train_imgs_labels

    test_imgs = imgs[train_num::, :]
    test_imgs_labels = labels[train_num::]

    test_texts = texts[train_num::, :]
    test_texts_labels = test_imgs_labels

    data = [[[train_imgs, train_imgs_labels], [train_imgs, train_imgs_labels], [test_imgs, test_imgs_labels]],
                [[train_texts, train_texts_labels], [train_texts, train_texts_labels], [test_texts, test_texts_labels]]]
    return data, ret_input_size


def load_nus_mnist_feature(inx=None):
    train_num = 20000
    valid_num = 8844

    nus_mnist = h5py.File('./data/nus_mnist120X120.h5')
    nus_imgs = nus_mnist['nus_Bow_int'][:].astype('float32')
    nus_tags = nus_mnist['nus_AllTags1k'][:].astype('float32')
    mnist_view1 = nus_mnist['mnist_view1'][:].astype('float32')
    mnist_view2 = nus_mnist['mnist_view2'][:].astype('float32')
    labels = nus_mnist['labels_all_views'][:].astype('float32').reshape([-1])

    ret_input_size = [nus_imgs.shape[1::], nus_tags.shape[1::], mnist_view1.shape[1::], mnist_view2.shape[1::]]
    if inx is None:
        inx = np.arange(nus_imgs.shape[0])
        np.random.shuffle(inx)
    train_inx = inx[0: train_num]
    valid_inx = inx[train_num: train_num + valid_num]
    test_inx = inx[train_num + valid_num::]

    train_imgs = nus_imgs[train_inx, :]
    train_imgs_labels = labels[train_inx]
    train_tags = nus_tags[train_inx, :]
    train_tags_labels = labels[train_inx]
    train_mnist_view1 = mnist_view1[train_inx, :]
    train_mnist_view1_labels = labels[train_inx]
    train_mnist_view2 = mnist_view2[train_inx, :]
    train_mnist_view2_labels = labels[train_inx]

    valid_imgs = nus_imgs[valid_inx, :]
    valid_imgs_labels = labels[valid_inx]
    valid_tags = nus_tags[valid_inx, :]
    valid_tags_labels = labels[valid_inx]
    valid_mnist_view1 = mnist_view1[valid_inx, :]
    valid_mnist_view1_labels = labels[valid_inx]
    valid_mnist_view2 = mnist_view2[valid_inx, :]
    valid_mnist_view2_labels = labels[valid_inx]

    test_imgs = nus_imgs[test_inx, :]
    test_imgs_labels = labels[test_inx]
    test_tags = nus_tags[test_inx]
    test_tags_labels = labels[test_inx]
    test_mnist_view1 = mnist_view1[test_inx, :]
    test_mnist_view1_labels = labels[test_inx]
    test_mnist_view2 = mnist_view2[test_inx, :]
    test_mnist_view2_labels = labels[test_inx]

    data = [[[train_imgs, train_imgs_labels], [valid_imgs, valid_imgs_labels], [test_imgs, test_imgs_labels]], [[train_tags, train_tags_labels], [valid_tags, valid_tags_labels], [test_tags, test_tags_labels]], [[train_mnist_view1, train_mnist_view1_labels], [valid_mnist_view1, valid_mnist_view1_labels], [test_mnist_view1, test_mnist_view1_labels]], [[train_mnist_view2, train_mnist_view2_labels], [valid_mnist_view2, valid_mnist_view2_labels], [test_mnist_view2, test_mnist_view2_labels]]]
    return data, ret_input_size


def load_raw_nus_mnist_feature(inx=None):
    train_num = 20000
    valid_num = 8844

    nus_mnist = h5py.File('./data/nus_mnist120X120.h5')
    nus_imgs = nus_mnist['raw_images'][:]
    nus_tags = nus_mnist['nus_AllTags1k'][:].astype('float32')
    mnist_view1 = nus_mnist['mnist_view1'][:].astype('float32').reshape([-1, 28, 28, 1])
    mnist_view2 = nus_mnist['mnist_view2'][:].astype('float32').reshape([-1, 28, 28, 1])
    labels = nus_mnist['labels_all_views'][:].astype('float32').reshape([-1])

    ret_input_size = [nus_imgs.shape[1::], nus_tags.shape[1::], mnist_view1.shape[1::], mnist_view2.shape[1::]]
    if inx is None:
        inx = np.arange(nus_imgs.shape[0])
        np.random.shuffle(inx)
    train_inx = inx[0: train_num]
    valid_inx = inx[train_num: train_num + valid_num]
    test_inx = inx[train_num + valid_num::]

    train_imgs = nus_imgs[train_inx, :]
    train_imgs_labels = labels[train_inx]
    train_tags = nus_tags[train_inx, :]
    train_tags_labels = labels[train_inx]
    train_mnist_view1 = mnist_view1[train_inx, :]
    train_mnist_view1_labels = labels[train_inx]
    train_mnist_view2 = mnist_view2[train_inx, :]
    train_mnist_view2_labels = labels[train_inx]

    valid_imgs = nus_imgs[valid_inx, :]
    valid_imgs_labels = labels[valid_inx]
    valid_tags = nus_tags[valid_inx, :]
    valid_tags_labels = labels[valid_inx]
    valid_mnist_view1 = mnist_view1[valid_inx, :]
    valid_mnist_view1_labels = labels[valid_inx]
    valid_mnist_view2 = mnist_view2[valid_inx, :]
    valid_mnist_view2_labels = labels[valid_inx]

    test_imgs = nus_imgs[test_inx, :]
    test_imgs_labels = labels[test_inx]
    test_tags = nus_tags[test_inx]
    test_tags_labels = labels[test_inx]
    test_mnist_view1 = mnist_view1[test_inx, :]
    test_mnist_view1_labels = labels[test_inx]
    test_mnist_view2 = mnist_view2[test_inx, :]
    test_mnist_view2_labels = labels[test_inx]

    data = [[[train_imgs, train_imgs_labels], [valid_imgs, valid_imgs_labels], [test_imgs, test_imgs_labels]], [[train_tags, train_tags_labels], [valid_tags, valid_tags_labels], [test_tags, test_tags_labels]], [[train_mnist_view1, train_mnist_view1_labels], [valid_mnist_view1, valid_mnist_view1_labels], [test_mnist_view1, test_mnist_view1_labels]], [[train_mnist_view2, train_mnist_view2_labels], [valid_mnist_view2, valid_mnist_view2_labels], [test_mnist_view2, test_mnist_view2_labels]]]
    return data, ret_input_size


def load_mnist(inx=None, D=1):
    mnist = sio.loadmat('./data/mnist/mnist.mat')
    if inx is None:
        inx = np.arange(mnist['imgs'].shape[0])
        np.random.shuffle(inx)

    imgs = mnist['imgs'].astype('float32')
    labels = mnist['labels'].reshape([-1])
    train_num = 30000
    valid_num = 10000

    if D == 2:
        imgs = imgs.reshape([-1, 28, 28, 1], order='F')
        # imgs = np.transpose(imgs, axes=[0, 2, 1, 3])
        mnist_view1 = imgs[:, :, 0:14, :].reshape([-1, 28, 14, 1])
        mnist_view2 = imgs[:, :, 14::, :].reshape([-1, 28, 14, 1])
    else:
        mnist_view1 = imgs[:, 0: 392]
        mnist_view2 = imgs[:, 392::]


    ret_input_size = [mnist_view1.shape[1::], mnist_view2.shape[1::]]

    train_inx = inx[0: train_num]
    valid_inx = inx[train_num: train_num + valid_num]
    test_inx = inx[train_num + valid_num::]

    train_mnist_view1 = mnist_view1[train_inx, :]
    train_mnist_view1_labels = labels[train_inx]
    train_mnist_view2 = mnist_view2[train_inx, :]
    train_mnist_view2_labels = labels[train_inx]


    valid_mnist_view1 = mnist_view1[valid_inx, :]
    valid_mnist_view1_labels = labels[valid_inx]
    valid_mnist_view2 = mnist_view2[valid_inx, :]
    valid_mnist_view2_labels = labels[valid_inx]


    test_mnist_view1 = mnist_view1[test_inx, :]
    test_mnist_view1_labels = labels[test_inx]
    test_mnist_view2 = mnist_view2[test_inx, :]
    test_mnist_view2_labels = labels[test_inx]

    data = [[[train_mnist_view1, train_mnist_view1_labels], [valid_mnist_view1, valid_mnist_view1_labels], [test_mnist_view1, test_mnist_view1_labels]], [[train_mnist_view2, train_mnist_view2_labels], [valid_mnist_view2, valid_mnist_view2_labels], [test_mnist_view2, test_mnist_view2_labels]]]
    return data, ret_input_size


def load_noisyMNIST(D=1):
    mnist = sio.loadmat('./data/mnist/noisy_MNIST.mat')

    train_mnist_view1 = mnist['X1'].astype('float32')
    train_mnist_view1_labels = mnist['trainLabel']
    train_mnist_view2 =mnist['X2'].astype('float32')
    train_mnist_view2_labels = mnist['trainLabel']


    valid_mnist_view1 = mnist['XV1'].astype('float32')
    valid_mnist_view1_labels = mnist['tuneLabel']
    valid_mnist_view2 = mnist['XV2'].astype('float32')
    valid_mnist_view2_labels = mnist['tuneLabel']


    test_mnist_view1 = mnist['XTe1'].astype('float32')
    test_mnist_view1_labels = mnist['testLabel']
    test_mnist_view2 = mnist['XTe2'].astype('float32')
    test_mnist_view2_labels = mnist['testLabel']

    if D == 2:
        train_mnist_view1 = train_mnist_view1.reshape([-1, 28, 28, 1], order='F')
        train_mnist_view2 = train_mnist_view2.reshape([-1, 28, 28, 1], order='F')
        valid_mnist_view1 = valid_mnist_view1.reshape([-1, 28, 28, 1], order='F')
        valid_mnist_view2 = valid_mnist_view2.reshape([-1, 28, 28, 1], order='F')
        test_mnist_view1 = test_mnist_view1.reshape([-1, 28, 28, 1], order='F')
        test_mnist_view2 = test_mnist_view2.reshape([-1, 28, 28, 1], order='F')

    ret_input_size = [train_mnist_view1.shape[1::], train_mnist_view2.shape[1::]]

    data = [[[train_mnist_view1, train_mnist_view1_labels], [valid_mnist_view1, valid_mnist_view1_labels], [test_mnist_view1, test_mnist_view1_labels]], [[train_mnist_view2, train_mnist_view2_labels], [valid_mnist_view2, valid_mnist_view2_labels], [test_mnist_view2, test_mnist_view2_labels]]]
    return data, ret_input_size

def load_nMSAD(inx_num=0, D=1):
    SAD = sio.loadmat('./data/nMSAD/SAD.mat')['SAD']
    SAD_lab = sio.loadmat('./data/nMSAD/SAD_lab.mat')['SAD_lab']
    MV1 = sio.loadmat('./data/nMSAD/MV1.mat')['MV1']
    MV1_lab = sio.loadmat('./data/nMSAD/MV1_lab.mat')['MV1_lab']
    MV2 = sio.loadmat('./data/nMSAD/MV2.mat')['MV2']
    MV2_lab = sio.loadmat('./data/nMSAD/MV2_lab.mat')['MV2_lab']
    shuffle_inx = sio.loadmat('./data/nMSAD/shuffle_inx.mat')['shuffle_inx']


    shuffle_inx = shuffle_inx[inx_num] - 1
    data = [MV1, MV2, SAD]
    labels = [MV1_lab, MV2_lab, SAD_lab]

    train_num = 4000
    valid_num = 800

    train_inx = shuffle_inx[0: train_num]
    valid_inx = shuffle_inx[train_num: train_num + valid_num]
    test_inx = shuffle_inx[train_num + valid_num::]
    train, train_labels, valid, valid_labels, test, test_labels = [], [], [], [], [], []

    for i in range(len(data)):
        labels[i] = labels[i].reshape([-1])
        if D == 2:
            if (data[i].shape[1] % 28) == 0:
                dim = 28
            else:
                dim = 25
            train.append(data[i][train_inx].reshape([train_inx.shape[0], dim, -1, 1], order='F'))
            train_labels.append(labels[i][train_inx])

            valid.append(data[i][valid_inx].reshape([valid_inx.shape[0], dim, -1, 1], order='F'))
            valid_labels.append(labels[i][valid_inx])

            test.append(data[i][test_inx].reshape([test_inx.shape[0], dim, -1, 1], order='F'))
            test_labels.append(labels[i][test_inx])
        else:
            train.append(data[i][train_inx])
            train_labels.append(labels[i][train_inx])

            valid.append(data[i][valid_inx])
            valid_labels.append(labels[i][valid_inx])

            test.append(data[i][test_inx])
            test_labels.append(labels[i][test_inx])

    ret_input_size = [train[0].shape[1::], train[1].shape[1::], train[2].shape[1::]]

    data = [[[train[0], train_labels[0]], [valid[0], valid_labels[0]], [test[0], test_labels[0]]], [[train[1], train_labels[1]], [valid[1], valid_labels[1]], [test[1], test_labels[1]]], [[train[2], train_labels[2]], [valid[2], valid_labels[2]], [test[2], test_labels[2]]]]
    return data, ret_input_size
