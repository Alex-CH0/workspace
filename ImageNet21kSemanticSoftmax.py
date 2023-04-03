import tensorflow as tf
import numpy as np
import pickle


class ImageNet21kSemanticSoftmax:
    def __init__(self, **kwargs):
        self.args = kwargs
        with open(file='/home2/keeyoung/alex/NextVit/data.pkl', mode='rb') as f:
            self.tree = pickle.load(f)
        self.class_tree_list = self.tree['class_tree_list']
        self.class_names = np.array(list(self.tree['class_description'].values()))
        self.max_normalization_factor = 2e1
        num_classes = len(self.class_tree_list)
        temp_class_depth = []
        for i in range(num_classes):
            temp_class_depth.append(len(self.class_tree_list[i]) - 1)
        self.class_depth = tf.stack(temp_class_depth)
        max_depth = int(tf.reduce_max(self.class_depth).numpy())

        # process semantic relations
        hist_tree = tf.histogram_fixed_width(self.class_depth, value_range=[0, max_depth], nbins=max_depth + 1)
        ind_list = []
        class_names_ind_list = []
        hirarchy_level_list = []
        cls = tf.range(num_classes)
        for i in range(max_depth):
            if hist_tree[i] > 1:
                hirarchy_level_list.append(i)
                ind_list.append(tf.boolean_mask(cls, self.class_depth == i))
                class_names_ind_list.append(self.class_names[ind_list[-1].numpy()])
        self.hierarchy_indices_list = ind_list
        self.hirarchy_level_list = hirarchy_level_list
        self.class_names_ind_list = class_names_ind_list

        rows = []
        for item in self.class_tree_list:
            rows.append(np.pad(item, (0, 12), 'constant', constant_values=0)[:12])
        self.after_class_tree_list = np.concatenate(rows, axis=0).reshape(-1, 12)

        # calculating normalization array
        self.normalization_factor_list = tf.concat([tf.zeros_like(hist_tree)[:-1], [hist_tree[-1]]], axis=0)
        self.normalization_factor_list = tf.concat([tf.reduce_sum(hist_tree[i:]) for i in range(max_depth)], axis=0)
        self.normalization_factor_list = self.normalization_factor_list[0] / self.normalization_factor_list
        if self.max_normalization_factor:
            self.normalization_factor_list = tf.clip_by_value(self.normalization_factor_list,
                                                              clip_value_min=tf.float32.min,
                                                              clip_value_max=self.max_normalization_factor)

    def get_multiply_factor(self):
        multiply_factor = []
        for i, items in enumerate(self.after_class_tree_list):
            temp_list = []
            for j, item in enumerate(items):
                elm = 1 if item > 0 or (i == 0 and j == 0) else 0
                temp_list.append(elm)
            multiply_factor.append(temp_list)
        return np.array(multiply_factor)

    def split_logits_to_semantic_logits(self, logits):
        """
        split logits to 11 different hierarchies.
        :param self.self.hierarchy_indices_list: a list of size [num_of_hierarchies].
        Each element in the list is a tensor that contains the corresponding indices for the relevant hierarchy
        """
        semantic_logit_list = []
        for i, ind in enumerate(self.hierarchy_indices_list):
            logits_i = tf.gather(logits, ind, axis=1)
            semantic_logit_list.append(logits_i)
        return semantic_logit_list

    def convert_targets_to_semantic_targets(self, targets_original: tf.Tensor) -> tf.Tensor:

        targets = tf.identity(targets_original)  # dont edit original targets
        batch_size = tf.shape(targets)[0]
        # targets = tf.cast(targets, dtype=tf.int64)
        tensor_list = tf.TensorArray(tf.int64, size=batch_size)
        for i in range(batch_size):
            target = targets[i]
            cls_multi_list = tf.gather(self.after_class_tree_list, target)[0]
            tensor_list.write(i, cls_multi_list)

        semantic_targets_list = tf.stack(tensor_list.stack())
        return semantic_targets_list
