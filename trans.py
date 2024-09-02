import tensorflow as tf
from keras.src.layers import LayerNormalization, Dense, MultiHeadAttention
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    CompatibleExtractPatches,
    conv2d_no_bias,
    drop_block,
    layer_norm,
    mlp_block,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights


def LWA(
        inputs, kernel_size=7, num_heads=4, key_dim=0, out_weight=True, qkv_bias=True, out_bias=True, attn_dropout=0,
        output_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    out_shape = cc
    qkv_out = num_heads * key_dim

    should_pad_hh, should_pad_ww = max(0, kernel_size - hh), max(0, kernel_size - ww)
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])
        _, hh, ww, cc = inputs.shape

    qkv = keras.layers.Dense(qkv_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    query, key_value = tf.split(qkv, [qkv_out, qkv_out * 2], axis=-1)  # Matching weights from PyTorch
    query = tf.expand_dims(tf.reshape(query, [-1, hh * ww, num_heads, key_dim]),
                           -2)  # [batch, hh * ww, num_heads, 1, key_dim]

    # key_value: [batch, height // kernel_size, width // kernel_size, kernel_size, kernel_size, key + value]
    key_value = CompatibleExtractPatches(sizes=kernel_size, strides=1, padding="VALID", compressed=False)
    padded = (kernel_size - 1) // 2
    # torch.pad 'replicate'
    key_value = tf.concat(
        [tf.repeat(key_value[:, :1], padded, axis=1), key_value, tf.repeat(key_value[:, -1:], padded, axis=1)], axis=1)
    key_value = tf.concat(
        [tf.repeat(key_value[:, :, :1], padded, axis=2), key_value, tf.repeat(key_value[:, :, -1:], padded, axis=2)],
        axis=2)

    key_value = tf.reshape(key_value, [-1, kernel_size * kernel_size, key_value.shape[-1]])
    key, value = tf.split(key_value, 2,
                          axis=-1)  # [batch * block_height * block_width, kernel_size * kernel_size, key_dim]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]),
                       [0, 2, 3, 1])  # [batch * hh*ww, num_heads, key_dim, kernel_size * kernel_size]
    key = tf.reshape(key, [-1, hh * ww, num_heads, key_dim,
                           kernel_size * kernel_size])  # [batch, hh*ww, num_heads, key_dim, kernel_size * kernel_size]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 1, 3])
    value = tf.reshape(value, [-1, hh * ww, num_heads, kernel_size * kernel_size,
                               key_dim])  # [batch, hh*ww, num_heads, kernel_size * kernel_size, key_dim]
    # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }")

    # [batch, hh * ww, num_heads, 1, kernel_size * kernel_size]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    # attention_scores = MultiHeadRelativePositionalKernelBias(input_height=hh, name=name and name + "pos")(
    #    attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(
        attention_scores) if attn_dropout > 0 else attention_scores

    # attention_output = [batch, block_height * block_width, num_heads, 1, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.reshape(attention_output, [-1, hh, ww, num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if should_pad_hh or should_pad_ww:
        attention_output = attention_output[:, : hh - should_pad_hh, : ww - should_pad_ww, :]

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(
            attention_output)
    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(
        attention_output) if output_dropout > 0 else attention_output
    return attention_output


def LWA_block(inputs, attn_kernel_size=7, num_heads=4, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0,
              layer_scale=-1, name=None):
    input_channel = inputs.shape[-1]

    attn = layer_norm(inputs, name=name + "attn_")
    attn = LWA(attn, attn_kernel_size, num_heads, attn_dropout=attn_drop_rate, name=name + "attn_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale,
                         name=name + "1_gamma") if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = layer_norm(attn_out, name=name + "mlp_")
    mlp = mlp_block(mlp, int(input_channel * mlp_ratio), activation="gelu", name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale,
                        name=name + "2_gamma") if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, dff, num_transformer_blocks, mlp_units, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.dff = dff
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.rate = rate

    def build(self, input_shape):
        self.embedding_dim = input_shape[-1]
        self.multi_head_attention = MultiHeadAttention(num_heads=self.num_heads,
                                                       key_dim=self.embedding_dim // self.num_heads)
        self.dense1 = Dense(self.dff, activation='relu')
        self.dense2 = Dense(self.embedding_dim)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.layernorm4 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.rate)
        self.dropout2 = tf.keras.layers.Dropout(self.rate)
        self.dropout3 = tf.keras.layers.Dropout(self.rate)
        self.dropout4 = tf.keras.layers.Dropout(self.rate)
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs, training=False):
        x, mask = inputs

        for _ in range(self.num_transformer_blocks):
            # Multi-Head Self-Attention
            attn_output = self.multi_head_attention(query=x, value=x, key=x, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)

            # Position-wise Feed Forward
            ffn_output = self.dense2(self.dense1(out1))
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)

            x = out2

        # MLP Head
        mlp_output = self.dense2(self.dense1(x))
        mlp_output = self.dropout3(mlp_output, training=training)
        mlp_output = self.layernorm3(x + mlp_output)

        # Local Window Attention
        attn_output = LWA_block(x, attn_kernel_size=7, num_heads=4, mlp_ratio=4)
        attn_output = self.dropout4(attn_output, training=training)
        out4 = self.layernorm4(mlp_output + attn_output)

        return out4
