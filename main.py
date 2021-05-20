import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from Dataset import Dataset
import utils
from models.SVD import SVD as APT
from models.pure_SVD import SVD as PureSVD

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "ml-100k", "Choose a dataset.")
flags.DEFINE_string('path', 'Data/', 'Input data path.')
flags.DEFINE_string('gpu', '0', 'Input data path.')
flags.DEFINE_integer('verbose', 1, 'Evaluate per X epochs.')
flags.DEFINE_integer('batch_size', 4096, 'batch_size')
flags.DEFINE_integer('epochs', 25, 'Number of epochs.')
flags.DEFINE_integer('embed_size', 64, 'Embedding size.')
flags.DEFINE_integer('dns', 0, 'number of negative sample for each positive in dns.')
flags.DEFINE_integer('per_epochs', 1, 'pass')
flags.DEFINE_float('reg', 0.02, 'Regularization for user and item embeddings.')
flags.DEFINE_float('lr', 0.05, 'Learning rate.')
flags.DEFINE_bool('reg_data', True, 'Regularization for adversarial loss')
flags.DEFINE_string('rs', 'svd', 'recommender system')
flags.DEFINE_bool("is_train", True, "train online or load model")
flags.DEFINE_integer("top_k", 50, "top k")
flags.DEFINE_list("target_item", [1679], "attack target item")
flags.DEFINE_float("attack_size", 0.03, "attack size")
flags.DEFINE_string("attack_type", "GAN", "attack type")
flags.DEFINE_float("data_size", 1., "pass")
flags.DEFINE_integer('target_index', 0, 'select target items')
flags.DEFINE_integer('extend', 50, 'the number of ERM users')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu


def get_rs(rs, dataset, extend):
    if (rs == 'svd'):
        rs = APT(dataset.num_users, dataset.num_items, dataset, extend)
    elif (rs == 'puresvd'):
        rs = PureSVD(dataset.num_users, dataset.num_items, dataset)
    else:
        print("error")
        exit(0)
    return rs


if __name__ == '__main__':
    extend = 0

    # target items
    a = [[1485, 1320, 821, 1562, 1531],
         [1018, 946, 597, 575, 516],
         [3639, 3698, 3622, 3570, 3503],
         [1032, 3033, 2797, 2060, 1366],
         [1576, 926, 942, 848, 107],
         [539, 117, 1600, 1326, 208],
         [2504, 19779, 9624, 24064, 17390],
         [2417, 21817, 13064, 3348, 15085]]
    FLAGS.target_item = a[FLAGS.target_index]

    import time

    cur_time = time.strftime("%Y-%m-%d", time.localtime())
    # initialize dataset
    dataset = Dataset(FLAGS.path + FLAGS.dataset, FLAGS.reg_data)

    t_epochs = 30
    t1, t2, t3, t4, t5 = [], [], [], [], []
    for i in range(t_epochs):
        RS = get_rs("puresvd", dataset, extend)
        tf.reset_default_graph()
        RS.build_graph()
        print("Initialize %s" % FLAGS.rs)

        # start training
        test_hr, test_ndcg = RS.train(dataset, FLAGS.is_train, FLAGS.epochs, np.ones(dataset.num_users), False)
        # target item recommendation
        print("origin: target item: ", FLAGS.target_item)
        hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
        t1.append(hr)
        t2.append(ndcg)
        t3.append(ps)
        t4.append(test_hr)
        t5.append(test_ndcg)
    print("clean model target items:",np.mean(t1))

    # attack
    attack_size = int(dataset.full_num_users * FLAGS.attack_size)
    poison_user = np.load("poison_data/%s_%s_poisoning_%d.npy" % (
        FLAGS.dataset, FLAGS.attack_type, attack_size))
    temp_user = np.mean(dataset.trainMatrix.toarray(), axis=0, keepdims=True)
    temp_user = np.round(temp_user * dataset.max_rate) / dataset.max_rate
    # poison_user = np.concatenate([poison_user, temp_user], axis=0)
    dataset = utils.estimate_dataset(dataset, poison_user)
    print("the users after attack:", dataset.num_users)

    # after poisoning
    extend = FLAGS.extend
    t1, t2, t3, t4, t5 = [], [], [], [], []
    for i in range(t_epochs):
        print("cur ", i)
        RS = get_rs("puresvd", dataset, extend)
        tf.reset_default_graph()
        RS.build_graph()
        test_hr, test_ndcg = RS.train(dataset, True, FLAGS.epochs, np.ones(dataset.num_users), False)
        # target item recommendation
        print("after attack: target item: ", FLAGS.target_item)
        hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
        t1.append(hr)
        t2.append(ndcg)
        t3.append(ps)
        t4.append(test_hr)
        t5.append(test_ndcg)
    print("after attack target items:",np.mean(t1))

    extend = FLAGS.extend
    t1, t2, t3, t4, t5 = [], [], [], [], []
    for i in range(t_epochs):
        weight = np.ones(dataset.num_users + extend) * 0.23
        print("cur ", i)
        RS = get_rs("svd", dataset, extend)
        tf.reset_default_graph()
        RS.build_graph()
        test_hr, test_ndcg = RS.train(dataset, True, FLAGS.epochs, weight, False)
        print("after attack: target item: ", FLAGS.target_item)
        hr, ndcg, ps, rank = utils.recommend(RS, dataset, FLAGS.target_item, FLAGS.top_k)
        t1.append(hr)
        t2.append(ndcg)
        t3.append(ps)
        t4.append(test_hr)
        t5.append(test_ndcg)
    print("after defense target items:",np.mean(t1))
