import tensorflow as tf
from tensorflow.contrib import slim
import os
import numpy as np
import math
import time
from time import strftime
from time import localtime
import utils
import copy

flags = tf.flags
FLAGS = flags.FLAGS


class SVD:
    def __init__(self, num_users, num_items, dataset, extend):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = FLAGS.embed_size
        self.reg = 1e-12
        self.dataset = dataset
        self.coo_mx = self.dataset.trainMatrix.tocoo()
        self.mu_np = np.mean(self.coo_mx.data)
        self.mu = tf.constant(self.mu_np, shape=[], dtype=tf.float32)
        self.extend = extend
        self.samples = utils.sampling(self.dataset, 0)
        self.trainmatrix = self.dataset.trainMatrix.toarray()
        self.rate_mask = self.trainmatrix != 0
        self.type = 'adv'

    def create_placeholders(self):
        with tf.variable_scope('placeholder'):
            self.users_holder = tf.placeholder(tf.int32, shape=[None, 1], name='users')
            self.items_holder = tf.placeholder(tf.int32, shape=[None, 1], name='items')
            self.ratings_holder = tf.placeholder(tf.float32, shape=[None, 1], name='ratings')
            self.weight = tf.placeholder(tf.float32, name='weight')

    def create_user_terms(self):
        num_users = self.num_users
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('user'):
            self.user_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_users + self.extend, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.p_u = tf.reduce_sum(tf.nn.embedding_lookup(
                self.user_embeddings,
                self.users_holder,
                name='p_u'), axis=1)

    def create_item_terms(self):
        num_items = self.num_items
        num_factors = self.num_factors

        w_init = slim.xavier_initializer
        with tf.variable_scope('item'):
            self.item_embeddings = tf.get_variable(
                name='embedding',
                shape=[num_items, num_factors],
                initializer=w_init(), regularizer=slim.l2_regularizer(self.reg))
            self.q_i = tf.reduce_sum(tf.nn.embedding_lookup(
                self.item_embeddings,
                self.items_holder,
                name='q_i'), axis=1)

    def create_prediction(self):
        with tf.variable_scope('prediction'):
            pred = tf.reduce_sum(tf.multiply(self.p_u, self.q_i), axis=1)
            self.pred = tf.expand_dims(pred, axis=-1)
            self.rate = tf.matmul(self.user_embeddings, tf.transpose(self.item_embeddings))
            self.rate_partial = tf.matmul(self.user_embeddings[:100], tf.transpose(self.item_embeddings))

    def create_optimizer(self):
        with tf.variable_scope('loss'):
            self.loss_pre = tf.reduce_mean(tf.pow(self.ratings_holder - self.pred, 2))
            self.loss_pre1 = tf.reduce_mean(tf.sigmoid(tf.pow(self.ratings_holder - self.pred, 2)))
            self.MAE = tf.reduce_mean(tf.abs(self.ratings_holder - self.pred))
            self.RMSE = tf.sqrt(tf.reduce_mean((self.ratings_holder - self.pred) * (self.ratings_holder - self.pred)))
            self.loss = tf.add(self.loss_pre,
                               (tf.reduce_mean(self.p_u * self.p_u) + tf.reduce_mean(self.q_i * self.q_i)) * self.reg,
                               name='loss')
            loss1 = tf.reduce_mean(tf.pow(self.ratings_holder - self.pred, 2))
            self.loss1 = tf.add(loss1,
                                tf.add_n(tf.get_collection(
                                    tf.GraphKeys.REGULARIZATION_LOSSES)))

            self.optimizer = tf.train.AdagradOptimizer(10.)
            if (FLAGS.dataset == 'yelp'):
                self.optimizer = tf.train.AdagradOptimizer(15.)
            self.train_op = self.optimizer.minimize(self.loss, name='optimizer')

    def build_graph(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.create_placeholders()
        self.create_user_terms()
        self.create_item_terms()
        self.create_prediction()
        self.create_optimizer()
        self.build_influence()

        self.sess.run(tf.global_variables_initializer())

    def train(self, dataset, is_train, nb_epochs, weight1, use_weight=True):
        ckpt_save_path = "pretrain/%s/%s/embed_%d/model_%s_%d" % (
            FLAGS.dataset, FLAGS.rs, FLAGS.embed_size, FLAGS.gpu, FLAGS.target_item[0])
        if (not os.path.exists(ckpt_save_path)):
            os.makedirs(ckpt_save_path)

        saver_ckpt = tf.train.Saver()

        # initialize test data for Evaluate
        samples = utils.sampling(dataset, 0)

        weight1 = (weight1 - np.min(weight1) + 1e-8) / (np.max(weight1) - np.min(weight1) + 1e-8)
        if (self.extend != 0):
            weight1[-self.extend:] = 0.5

        # don't use weights
        weight1 = np.ones_like(weight1)
        all_users = self.dataset.trainMatrix.toarray()
        ref_users_idx = utils.cal_neighbor(all_users, all_users, 1)[0]
        print(ref_users_idx)
        if (FLAGS.dataset == 'ml-100k'):
            ts = 200000.
            per_epochs = 2
            pre_training = 15
            select_num = 400
            up = 0.2
        elif (FLAGS.dataset == 'filmtrust'):
            ts = 400000.
            per_epochs = 3
            pre_training = 4
            select_num = 100
            up = 0.25
        elif (FLAGS.dataset == 'ml-1m'):
            ts = 200000.
            per_epochs = 2
            pre_training = 15
            select_num = 700
            up = 0.2
        elif (FLAGS.dataset == 'yelp'):
            ts = 100000.
            per_epochs = 2
            pre_training = 10
            select_num = 2500
            up = 0.2

        pre_influence = 0.
        for cur_epochs in range(nb_epochs):
            if (cur_epochs % per_epochs == pre_training % per_epochs and cur_epochs >= pre_training):
                if (self.type == 'adv'):
                    influence = self.influence_user(self.dataset, self.dataset.trainMatrix.toarray(), weight1)
                    inf_copy=influence.copy()
                    pos_idx = np.where(influence < 0)[0]
                    influence = influence[pos_idx]
                    # normalization
                    influence = ((influence - np.min(influence)) / (
                            np.max(influence) - np.min(influence)) - 1) * 0.0001
                    fake_users = utils.generate_fake(self.extend, self.dataset)
                    # fake_users += np.ones_like(fake_users) * up * inf_sign
                    fake_users += up
                    mask = np.zeros((self.extend, self.num_items))
                    inf_idx = np.argsort(inf_copy)
                    p = np.exp(-influence * ts) / np.sum(np.exp(-influence * ts))
                    print(p[0], np.max(p),np.min(p))
                    print(np.where(inf_idx == FLAGS.target_item[0])[0], np.where(inf_idx == FLAGS.target_item[1])[0],
                          np.where(inf_idx == FLAGS.target_item[2])[0], np.where(inf_idx == FLAGS.target_item[3])[0])
                    for kk in range(len(mask)):
                        iiidx = np.random.choice(pos_idx, select_num, False, p=p)
                        mask[kk, iiidx] = 1.
                else:
                    fake_users = utils.generate_fake(self.extend, self.dataset)
                    mask = np.zeros((self.extend, self.num_items))
                    for kk in range(len(mask)):
                        iiidx = np.random.choice(np.arange(self.num_items), select_num, False)
                        mask[kk, iiidx] = 1.
                fake_users *= mask
                fake_users = np.clip(np.round(fake_users * self.dataset.max_rate) / self.dataset.max_rate, 0, 1)
                dataset = utils.estimate_dataset(self.dataset, fake_users)
                print(np.sum(fake_users) / np.sum(fake_users != 0))
                print(np.sum(fake_users != 0) / len(fake_users))
                samples = utils.sampling(dataset, 0)
            batchs = utils.get_batchs(samples, FLAGS.batch_size)
            for i in range(len(batchs)):
                users, items, rates = batchs[i]
                weight_batch = weight1[users]
                feed_dict = {self.users_holder: users,
                             self.items_holder: items,
                             self.ratings_holder: rates,
                             self.weight: weight_batch}
                self.sess.run([self.train_op], feed_dict)
            if (cur_epochs % FLAGS.per_epochs == 0 or cur_epochs == nb_epochs - 1):
                if (FLAGS.dataset == 'yelp' and cur_epochs != nb_epochs - 1):
                    rate = self.sess.run(self.rate_partial)
                else:
                    rate = self.sess.run(self.rate)
                hr, ndcg = utils.train_evalute(rate, dataset, cur_epochs)
                # utils.save_model(ckpt_save_path, saver_ckpt, self.sess)
            else:
                print("cur epochs", cur_epochs)
        return hr, ndcg

    def get_embeddings(self):
        results = self.sess.run([self.rate, self.user_embeddings, self.item_embeddings])
        return results

    def influence_user(self, dataset, all_user, weight):
        import time
        s1 = time.time()
        scale = 1.
        i_epochs = 200
        batch_size = 16384
        if (FLAGS.dataset == 'filmtrust'):
            batch_size = 16384
            i_epochs = 1000
        if (FLAGS.dataset == 'ml-100k'):
            batch_size = 16384
            i_epochs = 1000
        if (FLAGS.dataset == 'ml-1m'):
            i_epochs = 200
            batch_size = 16384
        # IHVP
        start_time = time.time()
        feed_dict = {self.users_holder: self.samples[0],
                     self.items_holder: self.samples[1],
                     self.ratings_holder: self.samples[2]}
        test_val = self.sess.run(self.attack_grad, feed_dict)
        test_val = [self.convert_slice_to_dense(test_val[0]), self.convert_slice_to_dense(test_val[1])]
        cur_estimate = test_val.copy()
        feed1 = {place: cur for place, cur in zip(self.Test, test_val)}
        for j in range(i_epochs):
            feed2 = {place: cur for place, cur in zip(self.v_cur_est, cur_estimate)}
            r = np.random.choice(len(self.samples[0]), size=[batch_size], replace=False)
            users, items, rates = self.samples[0][r], self.samples[1][r], self.samples[2][r]
            feed_dict = {self.users_holder: users,
                         self.items_holder: items,
                         self.ratings_holder: rates}
            cur_estimate = self.sess.run(self.estimation_IHVP, feed_dict={**feed_dict, **feed1, **feed2})
        inverse_hvp1 = [b / scale for b in cur_estimate]
        duration = time.time() - start_time
        print('Inverse HVP by HVPs+Lissa: took %s minute %s sec' % (duration // 60, duration % 60))

        ii = utils.cal_neighbor(all_user, all_user, 10)
        val_lissa = 0
        for i in ii:
            cur_user = all_user[i]
            user = np.array([[i]], dtype=np.int)
            feed_dict = {self.users_holder: user,
                         self.ratings_holder: cur_user[:, None]}
            feed2 = {place: cur for place, cur in zip(self.v_cur_est, inverse_hvp1)}
            pert_vector_val = utils.pert_vector_product(self.per_loss, self.params, self.ratings_holder,
                                                                  self.v_cur_est, True)
            val_lissa1 = self.sess.run(pert_vector_val, feed_dict={**feed_dict, **feed2})
            val_lissa += -val_lissa1[0] - val_lissa1[1]
        return val_lissa[:, 0]

    def cal_near(self, influence, all_user, top_k):
        influence = (influence - np.min(influence)) / (np.max(influence) - np.min(influence))
        dis = np.linalg.norm(all_user - influence[None, :], axis=1)
        idx = np.argsort(dis)[:top_k]
        return idx

    def build_influence(self):
        with tf.variable_scope('influence'):
            self.params = [self.user_embeddings, self.item_embeddings]

            scale = 1.

            dty = tf.float32
            self.v_cur_est = [tf.placeholder(dty, shape=a.get_shape(), name="v_cur_est" + str(i)) for i, a in
                              enumerate(self.params)]
            self.Test = [tf.placeholder(dty, shape=a.get_shape(), name="test" + str(i)) for i, a in
                         enumerate(self.params)]

            hessian_vector_val = utils.hessian_vector_product(self.loss1, self.params, self.v_cur_est, True)
            self.estimation_IHVP = [g + cur_e - HV / scale
                                    for g, HV, cur_e in zip(self.Test, hessian_vector_val, self.v_cur_est)]

            attack_loss = tf.sigmoid(self.loss_pre)
            self.attack_grad = tf.gradients(attack_loss, self.params)
            per_rate = tf.matmul(self.p_u, tf.transpose(self.item_embeddings))
            per_loss1 = tf.reduce_mean(tf.pow(tf.transpose(self.ratings_holder) - per_rate, 2))
            self.per_loss = tf.add(per_loss1,
                                   tf.add_n(tf.get_collection(
                                       tf.GraphKeys.REGULARIZATION_LOSSES)))

    def convert_slice_to_dense(self, indexedslice):
        v = np.zeros(indexedslice.dense_shape)
        value = indexedslice[0]
        slice = indexedslice.indices
        for i in range(len(slice)):
            v[slice[i]] += value[i]
        return v
