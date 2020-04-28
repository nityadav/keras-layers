import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer


class TreeLSTM(Layer):
    """
    Tree-LSTM for binary tree. It needs a 2-tuple input, with input_shape being
    [(None, max_nodes, emb_dim), (None, max_nodes, 2)]
    """

    def __init__(self, units, max_nodes, **kwargs):
        if K.backend() != 'tensorflow':
            raise ValueError('TreeLSTM is currently supported only with Tensorflow backend.')
        self.units = units
        self.max_nodes = max_nodes
        self.supports_masking = True
        super(TreeLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Initialize all the parameters. TODO: use better bias, initializers, regularizers
        :param input_shape:
        :return:
        """
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        self.input_dim = input_shape[2]
        self.translation_kernel = self.add_weight(shape=(self.input_dim, self.units), initializer='normal',
                                                  name='translation_kernel')
        self.kernel = self.add_weight(shape=(self.units * 5, self.units), initializer='normal', name='kernel')
        self.recurrent_kernel = self.add_weight(shape=(self.units * 5, self.units * 2), initializer='normal',
                                                name='recurrent_kernel')
        self.bias = self.add_weight(shape=(self.units * 5, 1), initializer='zeros', name='bias')

        super(TreeLSTM, self).build(input_shape)

    def call(self, input_list, mask=None):
        """
        Uses a while loop (for scanning over tree-nodes) nested inside another while loop (for scannning over tree-
        samples) to compute hidden states of all the nodes for all the samples. tf.scan cannot be used because the
        number of leaves/nodes is not constant which makes tf.scan unable to infer shape during compile-time.
        :param input_list:
        :param mask:
        :return:
        """

        def _loop_over_samples_condition(sample_idx, _):
            return K.less(sample_idx, batch_size)

        def _loop_over_samples_body(sample_idx, batch_hidden_states):
            """
            Keeps concatenating hidden states for a tree-sample over hidden states for the batch
            :param sample_idx:
            :param batch_hidden_states:
            :return:
            """

            def _loop_over_nodes_condition(node_idx, h, c):
                return K.less(node_idx, num_nodes)

            def _loop_over_nodes_body(node_idx, sample_hidden_states, sample_memory_states):
                # get vectors for composition
                children_idxs = K.squeeze(K.gather(tree_struct, [node_idx]), 0)
                node_initial_state = K.gather(init_states, [node_idx])

                # compose operation for inner node
                node_hidden_state, node_memory_state = self.__compose__(node_initial_state,
                                                                        K.gather(sample_hidden_states, children_idxs),
                                                                        K.gather(sample_memory_states, children_idxs))

                # concatenate node hidden state with the rest of the hidden states of the tree-sample
                sample_hidden_states = K.concatenate([sample_hidden_states, node_hidden_state], axis=0)
                sample_memory_states = K.concatenate([sample_memory_states, node_memory_state], axis=0)
                return tf.add(node_idx, 1), sample_hidden_states, sample_memory_states

            init_states = K.squeeze(K.gather(translated_vecs, [sample_idx]), 0)
            tree_struct = K.squeeze(K.gather(child_vecs, [sample_idx]), 0)

            tree_info = K.squeeze(K.gather(num_vecs, [sample_idx]), 0)
            num_nodes = K.squeeze(K.gather(tree_info, [0]), 0)
            num_leaves = K.squeeze(K.gather(tree_info, [1]), 0)

            # prepare for the while loop over inner nodes
            sample_hidden_states = K.gather(init_states, K.arange(0, num_leaves))
            sample_memory_states = K.gather(init_states, K.arange(0, num_leaves))
            node_idx = num_leaves

            # while loop over inner nodes
            _, sample_hidden_states, sample_memory_states = tf.while_loop(_loop_over_nodes_condition,
                                                                          _loop_over_nodes_body,
                                                                          [node_idx, sample_hidden_states,
                                                                           sample_memory_states],
                                                                          shape_invariants=[node_idx.get_shape(),
                                                                                            tf.TensorShape(
                                                                                                [None, None]),
                                                                                            tf.TensorShape(
                                                                                                [None, None])])

            # pad the hidden states to max_nodes
            padding = [[0, self.max_nodes - num_nodes], [0, 0]]
            padded_hidden_states = tf.pad(sample_hidden_states, padding, "CONSTANT")

            # concatenate the hidden states for tree-sample with the batch hidden states
            batch_hidden_states = K.concatenate([batch_hidden_states, K.expand_dims(padded_hidden_states, axis=0)],
                                                axis=0)
            return tf.add(sample_idx, 1), batch_hidden_states

        # unpack the inputs
        initial_state = input_list[0]
        child_vecs = input_list[1]
        num_vecs = input_list[2]

        # translate the initial states
        translated_vecs = K.relu(K.dot(initial_state, self.translation_kernel))

        # prepare for the while loop over samples
        batch_size = K.squeeze(K.gather(K.shape(initial_state), [0]), 0)
        batch_hidden_states = K.constant(np.empty([0, self.max_nodes, self.units]))
        sample_idx = K.constant(value=0, dtype='int32', name='sample_idx')

        # while loop over samples
        _, batch_hidden_states = tf.while_loop(_loop_over_samples_condition,
                                               _loop_over_samples_body,
                                               [sample_idx, batch_hidden_states],
                                               shape_invariants=[sample_idx.get_shape(),
                                                                 tf.TensorShape([None, None, None])])

        return batch_hidden_states

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape[0], self.max_nodes, self.units

    def compute_mask(self, inputs, input_mask=None):
        """
        Compute the mask for the hidden state tensor. For each node, [True] indicates un-masked, and [False] indicates
        masked. The returned value has only one element in the last dimension, since feature dimension is reduced to a
        boolean
        :param inputs:
        :param input_mask:
        :return: A boolean tensor of shape (num_samples, num_nodes, 1)
        """
        return K.any(K.not_equal(inputs[0], 0.0), axis=-1)

    def get_config(self):
        model_config = {'units': self.units, 'max_nodes': self.max_nodes}
        base_config = super(TreeLSTM, self).get_config()
        return dict(list(base_config.items()) + list(model_config.items()))

    @staticmethod
    def __no_composition__(input_vec, _):
        """
        For testing purpose, whereby we return `input_vec` as it is. Classfier in the end of model then essentially
        classifies using the initial states
        :param input_vec:
        :return:
        """
        return input_vec

    def __compose__(self, initial_state, children_hidden_states, children_memory_states):
        flat_hidden = K.expand_dims(K.reshape(children_hidden_states, [-1]))
        flat_mem = K.expand_dims(K.reshape(children_memory_states, [-1]))
        z = K.dot(self.kernel, K.transpose(initial_state))
        z += K.dot(self.recurrent_kernel, flat_hidden)
        z += self.bias
        i = K.hard_sigmoid(z[:self.units, :])
        f = K.hard_sigmoid(z[self.units:self.units * 3, :])
        o = K.hard_sigmoid(z[self.units * 3:self.units * 4, :])
        u = K.tanh(z[self.units * 4:, :])
        c = K.sum(K.reshape(flat_mem * f, [self.units, 2]), axis=1, keepdims=True) + (i * u)
        h = o * K.tanh(c)
        return K.transpose(h), K.transpose(c)
