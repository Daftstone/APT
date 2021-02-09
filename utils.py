from time import time
from time import strftime
from time import localtime
import os
import tensorflow as tf
import numpy as np
import math
import copy
import scipy.sparse as sp

flags = tf.flags
FLAGS = flags.FLAGS


def generate_fake(n, dataset):
    poison_user = np.zeros((n, dataset.num_items))
    for j in range(dataset.num_items):
        t = np.clip(np.random.normal(dataset.distribution1[j][0], dataset.distribution1[j][1], n), 0, 1)
        poison_user[:, j] = np.round(t * dataset.max_rate) / dataset.max_rate
    return poison_user


def get_ckpt_path(is_attack=False):
    if (is_attack == False):
        time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
        ckpt_save_path = "pretrain/%s/%s/embed_%d/%s/model.ckpt" % (
            FLAGS.dataset, FLAGS.rs, FLAGS.embed_size, time_stamp)
        ckpt_restore_path = 0 if FLAGS.is_train == True else "pretrain/%s/%s/embed_%d/%s/model.ckpt" % (
            FLAGS.dataset, FLAGS.rs, FLAGS.embed_size, FLAGS.pretrain)
    else:
        time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
        ckpt_save_path = "pretrain/attack/%s/%s/embed_%d/%s/model.ckpt" % (
            FLAGS.dataset, FLAGS.rs, FLAGS.embed_size, time_stamp)
        ckpt_restore_path = 0 if is_attack == True else "pretrain/attack/%s/%s/embed_%d/%s/model.ckpt" % (
            FLAGS.dataset, FLAGS.rs, FLAGS.embed_size, FLAGS.pretrain)
    if (not os.path.exists(ckpt_save_path) and FLAGS.is_train == True):
        os.makedirs(ckpt_save_path)
    if (ckpt_restore_path and not os.path.exists(ckpt_restore_path)):
        os.makedirs(ckpt_restore_path)
    return ckpt_save_path, ckpt_restore_path


def load_model(ckpt_restore_path, saver_ckpt, sess):
    if (FLAGS.is_train == False and ckpt_restore_path != 0):
        saver_ckpt.restore(sess, ckpt_restore_path)
    else:
        print(FLAGS.is_train, ckpt_restore_path)
        print("Initialized from scratch")
    return sess


def save_model(ckpt_save_path, saver_ckpt, sess):
    try:
        saver_ckpt.save(sess, ckpt_save_path)
    except:
        print("model save error!")


def prepare_test(dataset, is_attack=False):
    feed_dicts = []
    for i in range(len(dataset.testRatings)):
        temp = evaluate_input(dataset, i, is_attack)
        feed_dicts.append(temp)
    return feed_dicts


def evaluate_input(dataset, user, is_attack):
    test_item = dataset.testRatings[user][1]
    item_input = dataset.testNegatives[user].copy()
    if (is_attack):
        test_item = FLAGS.target_item
    item_input.append(test_item)
    if (len(item_input) != 100):
        print(user)
        print(len(item_input))
    # assert len(item_input) == 100
    user_input = np.full(len(item_input), user, dtype='int32')[:, None]
    item_input = np.array(item_input)[:, None]
    return user_input, item_input


def sampling(dataset, num_neg, bpr=False):
    tt = dataset.trainMatrix.tocoo()
    user_input = np.array(tt.row)
    item_input = np.array(tt.col)
    rate_input = np.array(tt.data)
    t1, t2, t3 = [], [], []
    if (num_neg > 0):
        for i in range(dataset.num_users):
            if (len(dataset.allNegatives[i]) != 0):
                ll = int(len(dataset.trainList[i]) * num_neg)
                t1 += [i for ii in range(ll)]
                j = list(np.random.choice(dataset.allNegatives[i], ll, replace=False))
                t2 += j
                t3 += [0 for ii in range(ll)]
        if (bpr == False):
            user_input = np.concatenate([user_input, np.array(t1)], axis=0)
            item_input = np.concatenate([item_input, np.array(t2)], axis=0)
            rate_input = np.concatenate([rate_input, np.array(t3)], axis=0)
    user_input = user_input[:, None]
    item_input = item_input[:, None]
    rate_input = rate_input[:, None]
    neg_item_input = np.array(t2)[:, None]
    if (bpr == True):
        return [user_input, item_input, neg_item_input]
    else:
        return [user_input, item_input, rate_input]


def get_batchs(samples, batch_size):
    length = samples[0].shape[0]
    idx = np.arange(length)
    np.random.shuffle(idx)
    samples[0] = samples[0][idx]
    samples[1] = samples[1][idx]
    samples[2] = samples[2][idx]
    num = (length - 1) // batch_size + 1
    batchs = []
    for i in range(num):
        begin = i * batch_size
        end = i * batch_size + batch_size
        batchs.append([samples[0][begin:end], samples[1][begin:end], samples[2][begin:end]])
    return batchs


def recommend(model, dataset, target_item, _k):
    # target_test = prepare_target(dataset, target_item)
    # hr, ndcg = target_evaluate(model, dataset, target_test, _k)

    rate = model.sess.run(model.rate)[:dataset.num_users]
    user = dataset.trainMatrix.toarray()
    mask = user != 0

    ps = 0
    for j in target_item:
        ps += np.mean(rate[:dataset.origin_num_users, j])
    ps /= len(target_item)

    rate_copy = rate.copy()
    rate = rate * (1 - 99999 * mask)

    rank_list = np.zeros(dataset.origin_num_users)
    count = 0
    ndcg_count = 0
    print(dataset.origin_num_users)
    for i in range(dataset.origin_num_users):
        idx = np.argsort(rate[i])[::-1]
        rank_temp = 0
        for j in target_item:
            rank_temp += np.where(idx == j)[0][0]
            count += (j in idx[:_k])
            ndcg_count += math.log(2) / math.log(np.where(idx[:_k] == j)[0] + 2) if j in idx[:_k] else 0
        rank_list[i] = rank_temp / len(target_item)
    all_hr = count / dataset.origin_num_users / len(target_item)
    all_ndcg = ndcg_count / dataset.origin_num_users / len(target_item)
    print("recommend all user:", all_hr, all_ndcg)
    return all_hr, all_ndcg, ps, rank_list

def inter(a, b):
    return list(set(a) & set(b))


def prepare_target(dataset, target_item):
    def get_by_user(user):
        item_input = dataset.testNegatives[user].copy()
        item_input.append(target_item)
        assert len(item_input) == 100
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input

    feed_dicts = []
    for i in range(len(dataset.testRatings)):
        temp = get_by_user(i)
        feed_dicts.append(temp)
    return feed_dicts


def target_evaluate(model, dataset, target_test, _k):
    result = evaluate(model, dataset, target_test, _k)
    hr, ndcg, auc = result
    res = "HR = %.4f, NDCG = %.4f" % (hr, ndcg)
    print(res)
    return hr, ndcg


def evaluate(model, dataset, feed_dicts, _k):
    res = []
    for user in range(len(dataset.testRatings)):
        res.append(eval_by_user(user, feed_dicts, model, _k))
    res = np.array(res)
    hr, ndcg, auc = (res.mean(axis=0)).tolist()
    return hr, ndcg, auc


def eval_by_user(user, feed_dicts, model, _K=10):
    # get prredictions of data in testing set
    user_input, item_input = feed_dicts[user]
    feed_dict = get_test_feed_dicts(model, user_input, item_input)
    predictions = model.sess.run(feed_dict[0], feed_dict[1])

    neg_predict, pos_predict = predictions[:-1], predictions[-1]
    position = (neg_predict >= pos_predict).sum()

    # calculate from HR@1 to HR@100, and from NDCG@1 to NDCG@100, AUC
    hr = position < _K
    ndcg = math.log(2) / math.log(position + 2) if position < _K else 0
    auc = 1 - (position / len(neg_predict))  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]

    return hr, ndcg, auc


def get_test_feed_dicts(model, user_input, item_input):
    feed_dict = []
    if (FLAGS.rs == 'bpr'):
        feed_dict.append(model.output)
        feed_dict.append({model.user_input: user_input, model.item_input_pos: item_input})
    else:
        feed_dict.append(model.pred)
        feed_dict.append({model.users_holder: user_input, model.items_holder: item_input})
    return feed_dict


def estimate_dataset(dataset, initial_data):
    new_dataset = copy.deepcopy(dataset)
    for i in range(initial_data.shape[0]):
        item = []
        for j in range(initial_data.shape[1]):
            if (initial_data[i, j] != 0):
                item.append(j)
        new_dataset.trainList.append(item)
    csr_matrix = new_dataset.trainMatrix.tocsr()
    new_dataset.trainMatrix = sp.vstack([csr_matrix, sp.csr_matrix(initial_data)]).todok()
    new_dataset.num_users += initial_data.shape[0]
    if (FLAGS.dataset == 'filmtrust' or FLAGS.dataset == 'ml-100k'):
        new_dataset.allNegatives = new_dataset.load_all_negative(new_dataset.trainList)
    return new_dataset


def train_evalute(rate, dataset, cur_epochs):
    test_num = min(len(rate), dataset.origin_num_users)
    rate = rate[:test_num]
    user = dataset.trainMatrix.toarray()[:test_num]
    mask = user != 0
    rate[mask] = -np.inf
    count = 0
    for i in range(test_num):
        idx = np.argsort(rate[i])[::-1][:FLAGS.top_k]
        for j in FLAGS.target_item:
            count += (j in idx)
    all_hr = count / test_num / len(FLAGS.target_item)
    count = 0
    ndcg_count = 0
    for i in range(test_num):
        idx = np.argsort(rate[i])[::-1][:FLAGS.top_k]
        for j in [dataset.testRatings[i][1]]:
            count += (j in idx)
            ndcg_count += math.log(2) / math.log(np.where(idx == j)[0] + 2) if j in idx else 0
    all_hr1 = count / test_num
    ndcg_count /= test_num
    rmse = 0
    ps = 0
    for i in range(test_num):
        uu, ii, rr = dataset.testRatings[i]
        rmse += (rr * dataset.max_rate - rate[uu, ii] * dataset.max_rate) * (
                rr * dataset.max_rate - rate[uu, ii] * dataset.max_rate)
        ps += np.abs(rate[uu, ii] - rr) * dataset.max_rate
    rmse /= test_num
    ps /= test_num
    print("epochs %d: %.4f %.4f %.4f %.4f" % (cur_epochs, all_hr, all_hr1, rmse, ps))
    return all_hr1, ndcg_count


def cal_neighbor(group, all_user, top_k):
    dis = np.linalg.norm(all_user, axis=1)
    # dis = np.sum(all_user != 0, axis=1)
    idx = np.argsort(dis)[:top_k]
    # idx = [len(dis) - 1]
    # print("idx", idx)
    return idx

def pert_vector_product(ys, xs1, xs2, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs1)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs1)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs2)[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs2)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(xs2) \
                    for grad_elem in grads_with_none]
    return return_grads


def hessian_vector_product(ys, xs, v, do_not_sum_up=True):
    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")
    # First backprop
    grads = tf.gradients(ys, xs)

    # grads = xs
    assert len(grads) == length
    elemwise_products = [
        tf.multiply(grad_elem, tf.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]
    # Second backprop
    if do_not_sum_up:
        seperate = []
        for i in range(length):
            seperate.append(tf.gradients(elemwise_products[i], xs[i])[0])
        grads_with_none = seperate
    else:
        grads_with_none = tf.gradients(elemwise_products, xs)

    return_grads = [grad_elem if grad_elem is not None \
                        else tf.zeros_like(x) \
                    for x, grad_elem in zip(xs, grads_with_none)]
    return return_grads