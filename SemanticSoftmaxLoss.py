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
            
            
            log_preds = tf.math.log_softmax(logits_i, axis=1)

            targets_i_valid = tf.identity(targets_i)
            targets_i_valid = tf.where(targets_i_valid < 0, tf.constant(0, dtype=targets_i_valid.dtype),
                                       targets_i_valid)

            num_classes = logits_i.shape[-1]
            targets_classes = tf.one_hot(targets_i_valid, depth=num_classes, dtype=logits_i.dtype)
            targets_classes = targets_classes * (1 - self.label_smooth) + self.label_smooth / num_classes
            
            cross_entropy_loss_tot = -targets_classes * log_preds
            
            #masked = tf.boolean_mask(targets_i, tf.math.greater_equal(targets_i, tf.constant(0, dtype=targets_i.dtype)))
            temp_expand = tf.cast(tf.expand_dims(targets_i >= 0, axis=1), dtype=tf.float32)
            
            cross_entropy_loss_tot *= temp_expand
            cross_entropy_loss = tf.reduce_sum(cross_entropy_loss_tot, axis=-1)  # sum over classes
            loss_i = tf.reduce_mean(cross_entropy_loss)  # mean over batch
            
            losses_list.append(loss_i)


        total_sum = 0.
        for i, loss_h in enumerate(losses_list):  # summing over hirarchies
            temp = loss_h * tf.cast(self.semantic_softmax_processor.normalization_factor_list[i], dtype=tf.float32)
            total_sum += temp

        return total_sum
