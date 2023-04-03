import tensorflow as tf


class SemanticSoftmaxLoss(tf.keras.losses.Loss):
    def __init__(self, semantic_softmax_processor):
        super(SemanticSoftmaxLoss, self).__init__()
        self.semantic_softmax_processor = semantic_softmax_processor
        self.args = semantic_softmax_processor.args
        self.training = True
        self.label_smooth = self.args['label_smooth']

    def call(self, targets, logits):
        """
        Calculates the semantic cross-entropy loss distance between logits and targets
        """

        if not self.training:
            return 0

        semantic_logit_list = self.semantic_softmax_processor.split_logits_to_semantic_logits(logits)
        semantic_targets_tensor = self.semantic_softmax_processor.convert_targets_to_semantic_targets(targets)

        losses_list = []
        # scanning hirarchy_level_list
        for i in range(len(semantic_logit_list)):
            logits_i = semantic_logit_list[i]
            targets_i = semantic_targets_tensor[:, i]

            targets_i_valid = tf.identity(targets_i)
            targets_i_valid = tf.where(targets_i_valid < 0, tf.constant(0, dtype=targets_i_valid.dtype),
                                       targets_i_valid)

            num_classes = logits_i.shape[-1]
            targets_classes = tf.one_hot(targets_i_valid, depth=num_classes, dtype=logits_i.dtype)
            targets_classes = targets_classes * (1 - self.label_smooth) + self.label_smooth / num_classes

            loss_i = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)(logits_i,
                                                                                                      targets_classes)
            losses_list.append(loss_i)

        total_sum = 0.
        multiply_factor = self.semantic_softmax_processor.get_multiply_factor()
        for i, loss_h in enumerate(losses_list):  # summing over hirarchies
            mul_factor_i = multiply_factor[i]
            loss_h = tf.math.multiply(loss_h, mul_factor_i)
            temp = loss_h * tf.cast(self.semantic_softmax_processor.normalization_factor_list[i], dtype=tf.float32)
            total_sum += temp

        return total_sum
