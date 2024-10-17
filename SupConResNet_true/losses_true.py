import tensorflow as tf


class SupConLoss(tf.keras.losses.Loss):

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, name=None):
        super(SupConLoss, self).__init__(name=name)
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    @tf.function
    def call(self, y_true, y_pred):
        bsz = tf.shape(y_pred)[0] // 2
        f1, f2 = tf.split(y_pred, [bsz, bsz], axis=0)
        # 增加一个维度
        f1 = tf.expand_dims(f1, axis=1)
        f2 = tf.expand_dims(f2, axis=1)
        features = tf.concat([f1, f2], axis=1)
        labels, _ = tf.split(y_true, [bsz, bsz], axis=0)
        features = tf.convert_to_tensor(features)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = tf.reshape(features, [features.shape[0], features.shape[1], -1])

        batch_size = tf.shape(features)[0]
        labels = tf.reshape(labels, [-1, 1])
        expected_batch_size = tf.shape(labels)[0]
        if expected_batch_size != batch_size:
            tf.print(expected_batch_size,batch_size)

        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), dtype=tf.float32)
        contrast_count = features.shape[1]
        contrast_feature = tf.concat(tf.unstack(features, axis=1), axis=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        tf.debugging.check_numerics(anchor_feature, "anchor_feature NaN or Inf found")
        tf.debugging.check_numerics(contrast_feature, "contrast_feature NaN or Inf found")
        # Compute logits
        anchor_dot_contrast = tf.divide(
            tf.matmul(anchor_feature, contrast_feature, transpose_b=True),
            self.temperature)
        tf.debugging.check_numerics(anchor_dot_contrast, "anchor_dot_contrast NaN or Inf found")
        # For numerical stability
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        tf.debugging.check_numerics(logits_max, "logits_max NaN or Inf found")
        logits = anchor_dot_contrast - logits_max
        tf.debugging.check_numerics(logits, "Logits NaN or Inf found")
        # Tile mask
        mask = tf.tile(mask, [anchor_count, contrast_count])

        # Mask-out self-contrast cases
        logits_mask = tf.ones_like(mask) - tf.linalg.tensor_diag(tf.ones([batch_size * anchor_count]))
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        tf.debugging.check_numerics(exp_logits, "exp_logits NaN or Inf found")
        # Compute mean of log-likelihood over positive
        mask_pos_pairs = tf.reduce_sum(mask, axis=1)
        mask_pos_pairs = tf.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_pos_pairs

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))

        return loss
