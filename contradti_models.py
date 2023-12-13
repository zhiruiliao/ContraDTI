import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score as skauc
from sklearn.metrics import auc, precision_recall_curve


class SMILESConvEncoder(keras.Model):
    def __init__(self, d_model, smiles_vocab_size=35, max_smiles_len=80, kernel_size=7, **kwargs):
        super(SMILESConvEncoder, self).__init__(**kwargs)
        self.embed_layer = layers.Embedding(
                               input_dim=smiles_vocab_size, 
                               output_dim=d_model, 
                               input_length=max_smiles_len
                               )
        self.d_model = d_model
        self.smiles_vocab_size = smiles_vocab_size
        self.max_smiles_len = max_smiles_len
        self.dropout_layer = layers.Dropout(0.1)
        self.conv1 = layers.Conv1D(filters=d_model, kernel_size=kernel_size, activation='relu', padding='same')
        self.conv2 = layers.Conv1D(filters=d_model, kernel_size=kernel_size, activation='relu', padding='same')
        self.pooling_layer = layers.GlobalMaxPooling1D()

    def get_config(self):
        config = {
            "d_model": self.d_model,
            "smiles_vocab_size": self.smiles_vocab_size,
            "max_smiles_len": self.max_smiles_len,
            "kernel_size": self.kernel_size
            }
        base_config = super(SMILESConvEncoder, self).get_config()
        config.update(base_config)
        return config
    
    def call(self, inputs, training):
        x = inputs
        h = self.embed_layer(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.dropout_layer(h, training=training)
        h = self.pooling_layer(h)
        return h


class GraphConv(layers.Layer):
    """Graph convolution layer.
    Xnew = activation(AXW + b) * Mask
        Args:
            d_model: int, the output dimension.
            use_bias: bool, whether the bias is used.
            activation: str or callable, the activation function.

        Inputs:
            a: Adjacency matrix A. shape = `(batch_size, n, n)`
            x: Input matrix X. shape = `(batch_size, n, d_input)`
            mask: Mask. shape = `(batch_size, n)`

        Outputs:
            xnew: Updated feature matrix X_{i+1}. shape = `(batch_size, n, d_model)`
    """

    def __init__(self, d_model, use_bias=True, activation=None, **kwargs):
        super(GraphConv, self).__init__(**kwargs)
        self.d_model = d_model
        self.use_bias = use_bias
        self.activation = activation
        self.dense = layers.Dense(units=d_model, activation=activation, use_bias=use_bias)

    def get_config(self):
        config = {
            "d_model": self.d_model,
            "use_bias": self.use_bias,
            "activation": self.activation
        }
        base_config = super(GraphConv, self).get_config()
        config.update(base_config)
        return config

    def call(self, a, x, mask):
        ax = tf.matmul(a, x)
        z = self.dense(ax)
        return z * mask[:, :, tf.newaxis]


class GraphEncoderModel(keras.Model):
    """Two-layer graph convolution encoder followed by max pooling.
    H = relu(AXW_1 + b_1) * Mask
    Z = (A(H)W2 + b2) * Mask
        Args:
            d_model: int, the output dimension.
            dff: int, the middle dimension, i.e. W_1's dimension.
            use_bias: bool, whether the bias is used.
            activation: str or callable, the activation function.

        Inputs:
            data: (a, x, mask)
                a: Adjacency matrix A. shape = `(batch_size, n, n)`
                x: Input matrix X. shape = `(batch_size, n, d_input)`
                mask: Mask. shape = `(batch_size, n)`
            training: bool
        Outputs:
            z: Encoding vector. shape = `(batch_size, d_model)`
    """

    def __init__(self, d_model, dff, use_bias=True, dropout_rate=0.1, **kwargs):
        super(GraphEncoderModel, self).__init__(**kwargs)
        self.gcn_layer_1 = GraphConv(dff, use_bias=use_bias, activation='relu')
        self.gcn_layer_2 = GraphConv(d_model, use_bias=use_bias)
        self.dropout_layer = layers.Dropout(dropout_rate)
        self.pooling_layer = layers.GlobalMaxPooling1D()
        self.d_model = d_model
        self.dff = dff
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'dff': self.dff,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate
        }
        return config

    def call(self, data, training):
        a, x, mask = data
        h = self.gcn_layer_1(a, x, mask)
        z = self.gcn_layer_2(a, h, mask)
        z = self.dropout_layer(z, training=training)
        reverse_mask = tf.cast(tf.math.equal(mask, 0), tf.float32)[:, :, tf.newaxis]
        z += (reverse_mask * -1e9)
        z = self.pooling_layer(z)
        return z


class ContrastiveModel(keras.Model):
    def __init__(self, encoder, encoder_s, d_model, dff, num_classes, temperature, lmd, **kwargs):
        
        super(ContrastiveModel, self).__init__(**kwargs)
        
        self.temperature = temperature
        self.lmd = lmd
        self.num_classes = num_classes
        
        self.encoder = encoder
        self.encoder_s = encoder_s
        # Non-linear MLP as projection head
        
        self.s_head = keras.Sequential(
            [
                keras.Input(shape=(d_model,)),
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model)
            ],
            name="s_head"
        )
        
        self.g_head = keras.Sequential(
            [
                keras.Input(shape=(d_model,)),
                layers.Dense(dff, activation="relu"),
                layers.Dense(d_model)
            ],
            name="g_head"
        )
        # Single dense layer for linear probing
        self.linear_probe = keras.Sequential(
            [layers.Input(shape=(d_model,)), layers.Dense(num_classes)], name="linear_probe"
        )

        self.encoder.summary()
        self.encoder_s.summary()
        
        self.s_head.summary()
        self.g_head.summary()
        self.linear_probe.summary()
    
    def get_positive_probe(self, class_logits):
        probe = tf.nn.softmax(class_logits, axis=1)
        return probe[:, 1]
    
    def compile(self, contrastive_optimizer, probe_optimizer, finetune_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer
        self.finetune_optimizer = finetune_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.alignment_loss_tracker = keras.metrics.Mean(name='a_loss')
        
        self.contrastive_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        self.probe_loss_tracker = keras.metrics.Mean(name="p_loss")
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(name="p_acc")
        self.probe_auroc = keras.metrics.AUC(curve='ROC', name='p_auroc')
        self.probe_aupr = keras.metrics.AUC(curve='PR', name='p_aupr')
        self.mse = tf.keras.losses.MeanSquaredError()
        self.total_cl_loss_tracker = keras.metrics.Mean(name="t_loss")
    
    @property
    def metrics(self):
        return [
            self.alignment_loss_tracker,
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            self.total_cl_loss_tracker,
            self.probe_loss_tracker,
            self.probe_accuracy,
            self.probe_auroc,
            self.probe_aupr
        ]
    
    def alignment_loss(self, projections_smiles, projections_graph):
        projections_smiles = tf.math.l2_normalize(projections_smiles, axis=1)
        projections_graph = tf.math.l2_normalize(projections_graph, axis=1)
        
        similarities_sg = tf.matmul(projections_smiles, projections_graph, transpose_b=True)
        
        _loss = -similarities_sg
        return tf.reduce_mean(_loss)
        
    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.one_hot(tf.range(batch_size), batch_size * 2) # (n, 2n)
        masks = tf.one_hot(tf.range(batch_size), batch_size)
        
        similarities_11 = tf.matmul(projections_1, projections_1, transpose_b=True) / self.temperature
        similarities_11 = similarities_11 - masks * 1e9
        
        similarities_22 = tf.matmul(projections_2, projections_2, transpose_b=True) / self.temperature
        similarities_22 = similarities_22 - masks * 1e9
        
        similarities_12 = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )
        similarities_21 = tf.transpose(similarities_12)
        
        similarities_1 = tf.concat([similarities_12, similarities_11], 1)
        loss_1 = tf.nn.softmax_cross_entropy_with_logits(contrastive_labels, similarities_1)
        
        similarities_2 = tf.concat([similarities_21, similarities_22], 1)
        loss_2 = tf.nn.softmax_cross_entropy_with_logits(contrastive_labels, similarities_2)
        
        loss = tf.reduce_mean(loss_1 + loss_2)
        
        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        contrastive_labels_sparse = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels_sparse, similarities_1)
        self.contrastive_accuracy.update_state(contrastive_labels_sparse, similarities_2)

        return loss
    
    def train_step(self, data):
        (labels, real, fake, smiles) = data
        with tf.GradientTape() as tape:
            features_1 = self.encoder(real, training=True)
            features_2 = self.encoder(fake, training=True)
            features_smiles = self.encoder_s(smiles, training=True)
            
            # The representations are passed through a projection mlp
            projections_1 = self.g_head(features_1)
            projections_2 = self.g_head(features_2)
            projections_smiles = self.s_head(features_smiles)
            
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
            alignment_loss = self.alignment_loss(projections_smiles, projections_1)
            total_cl_loss = contrastive_loss + self.lmd * alignment_loss
            
        gradients = tape.gradient(
            total_cl_loss,
            self.encoder.trainable_weights + self.encoder_s.trainable_weights + self.g_head.trainable_weights + self.s_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.encoder_s.trainable_weights + self.g_head.trainable_weights + self.s_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.alignment_loss_tracker.update_state(alignment_loss)
        self.total_cl_loss_tracker.update_state(total_cl_loss)
        
        # Labels are only used in evalutation for an on-the-fly logistic regression
        with tf.GradientTape() as tape:
            features = self.encoder(real, training=False)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        self.probe_auroc.update_state(labels, self.get_positive_probe(class_logits))
        self.probe_aupr.update_state(labels, self.get_positive_probe(class_logits))
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        (labels, real, fake, smiles) = data
        features = self.encoder(real, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        self.probe_auroc.update_state(labels, self.get_positive_probe(class_logits))
        self.probe_aupr.update_state(labels, self.get_positive_probe(class_logits))
        # Only the probe metrics are logged at test time
        return {m.name: m.result() for m in self.metrics[2:]}
    
    def predict_step(self, data):
        features = self.encoder(data, training=False)
        class_logits = self.linear_probe(features, training=False)
        return self.get_positive_probe(class_logits)
    
    def call(self, data, training):
        return data
    
    def train_step_partially_labelled(self, unlabelled_data, labelled_data):
        (_not_used, real_unlabelled, fake_unlabelled, smiles_unlabelled) = unlabelled_data
        (labels, real_labelled, fake_labelled, smiles_labelled) = labelled_data
        
        (a_real_unlabelled, x_real_unlabelled, mask_real_unlabelled) = real_unlabelled
        (a_fake_unlabelled, x_fake_unlabelled, mask_fake_unlabelled) = fake_unlabelled
        
        (a_real_labelled, x_real_labelled, mask_real_labelled) = real_labelled
        (a_fake_labelled, x_fake_labelled, mask_fake_labelled) = fake_labelled
        
        a_real = tf.concat([a_real_unlabelled, a_real_labelled], axis=0)
        x_real = tf.concat([x_real_unlabelled, x_real_labelled], axis=0)
        mask_real = tf.concat([mask_real_unlabelled, mask_real_labelled], axis=0)
        
        a_fake = tf.concat([a_fake_unlabelled, a_fake_labelled], axis=0)
        x_fake = tf.concat([x_fake_unlabelled, x_fake_labelled], axis=0)
        mask_fake = tf.concat([mask_fake_unlabelled, mask_fake_labelled], axis=0)
        
        real = (a_real, x_real, mask_real)
        fake = (a_fake, x_fake, mask_fake)
        
        smiles = tf.concat([smiles_unlabelled, smiles_labelled], axis=0)
        
        with tf.GradientTape() as tape:
            features_1 = self.encoder(real, training=True)
            features_2 = self.encoder(fake, training=True)
            features_smiles = self.encoder_s(smiles, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.g_head(features_1)
            projections_2 = self.g_head(features_2)
            
            projections_smiles = self.s_head(features_smiles)
           
            
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
            alignment_loss = self.alignment_loss(projections_smiles, projections_1)
            
            total_cl_loss = contrastive_loss + self.lmd * alignment_loss
            
        gradients = tape.gradient(
            total_cl_loss,
            self.encoder.trainable_weights + self.encoder_s.trainable_weights + self.g_head.trainable_weights + self.s_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.encoder_s.trainable_weights + self.g_head.trainable_weights + self.s_head.trainable_weights,
            )
        )
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.alignment_loss_tracker.update_state(alignment_loss)
        self.total_cl_loss_tracker.update_state(total_cl_loss)
        
        # Labels are only used in evalutation for an on-the-fly logistic regression
        with tf.GradientTape() as tape:
            features = self.encoder(real_labelled, training=False)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        self.probe_auroc.update_state(labels, self.get_positive_probe(class_logits))
        self.probe_aupr.update_state(labels, self.get_positive_probe(class_logits))
        return {m.name: m.result() for m in self.metrics}
    
    def finetune_step(self, data):
        (labels, real, fake, smiles) = data
        with tf.GradientTape() as tape:
            features = self.encoder(real, training=True)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labels, class_logits)
        gradients = tape.gradient(
            probe_loss,
            self.encoder.trainable_weights + self.linear_probe.trainable_weights)
        self.finetune_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.linear_probe.trainable_weights)
        )
        self.probe_loss_tracker.update_state(probe_loss)
        self.probe_accuracy.update_state(labels, class_logits)
        self.probe_auroc.update_state(labels, self.get_positive_probe(class_logits))
        self.probe_aupr.update_state(labels, self.get_positive_probe(class_logits))
        return {m.name: m.result() for m in self.metrics}
