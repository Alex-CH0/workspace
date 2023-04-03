import tensorflow as tf

NORM_EPS = 1e-5
initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(tf.keras.layers.Layer):
    """Convolution + Batch Normalization + Relu layer"""

    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride):
        super(ConvBNReLU, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size,
                                           strides=stride, dtype=tf.float32, padding='same',
                                           groups=1, use_bias=False,
                                           kernel_initializer=initializer
                                           , bias_initializer='zeros'
                                           )
        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=NORM_EPS
        )
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config


class PatchEmbed(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        identity = tf.keras.layers.Lambda(
            lambda x: x
        )
        if stride == 2:
            self.avgpool = tf.keras.layers.AveragePooling2D((2, 2), strides=2)
            self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False,
                                               kernel_initializer=initializer
                                               , bias_initializer='zeros'
                                               )
            self.norm = tf.keras.layers.BatchNormalization(
                epsilon=NORM_EPS
            )
        elif in_channels != out_channels:
            self.avgpool = identity
            self.conv = tf.keras.layers.Conv2D(out_channels, kernel_size=1, strides=1, use_bias=False,
                                               kernel_initializer=initializer
                                               , bias_initializer='zeros'
                                               )
            self.norm = tf.keras.layers.BatchNormalization(
                epsilon=NORM_EPS
            )
        else:
            self.avgpool = identity
            self.conv = identity
            self.norm = identity

    def call(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv(x)
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config


class MHCA(tf.keras.layers.Layer):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        self.group_conv3x3 = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1,
                                                    padding='same', groups=out_channels // head_dim, use_bias=False,
                                                    kernel_initializer=initializer
                                                    , bias_initializer='zeros'
                                                    )
        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=NORM_EPS
        )
        self.activation = tf.keras.layers.ReLU()
        self.projection = tf.keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False,
                                                 kernel_initializer=initializer
                                                 , bias_initializer='zeros'
                                                 )

    def call(self, inputs):
        out = self.group_conv3x3(inputs)
        out = self.norm(out)
        out = self.activation(out)
        out = self.projection(out)
        return out

    def get_config(self):
        config = super().get_config()
        return config


# from https://keras.io/examples/vision/swin_transformers/
class DropPath(tf.keras.layers.Layer):
    """
    DropPath
    """

    def __init__(self, drop_prob=None, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output

    def get_config(self):
        config = super().get_config()
        return config


class Mlp(tf.keras.layers.Layer):
    """
    MLP
    """

    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., use_bias=True):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)

        self.conv1 = tf.keras.layers.Conv2D(hidden_dim, kernel_size=1, use_bias=use_bias,
                                            kernel_initializer=initializer
                                            , bias_initializer='zeros'
                                            )
        self.conv2 = tf.keras.layers.Conv2D(out_features, kernel_size=1, use_bias=use_bias,
                                            kernel_initializer=initializer
                                            , bias_initializer='zeros'
                                            )
        self.activation = tf.keras.layers.ReLU()
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.activation(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config


class NCB(tf.keras.layers.Layer):
    """
    Next Convolution Block
    """

    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(self.in_channels, self.out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = tf.keras.layers.BatchNormalization(
            epsilon=NORM_EPS
        )
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, use_bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)

    def call(self, inputs):
        x = self.patch_embed(inputs)
        x = x + self.attention_path_dropout(self.mhca(x))
        out = self.norm(x)
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x

    def get_config(self):
        config = super().get_config()
        return config


class E_MHSA(tf.keras.layers.Layer):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super(E_MHSA, self).__init__()

        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = tf.keras.layers.Dense(self.dim, use_bias=qkv_bias,
                                       kernel_initializer=initializer
                                       , bias_initializer='zeros'
                                       )
        self.k = tf.keras.layers.Dense(self.dim, use_bias=qkv_bias,
                                       kernel_initializer=initializer
                                       , bias_initializer='zeros'
                                       )
        self.v = tf.keras.layers.Dense(self.dim, use_bias=qkv_bias,
                                       kernel_initializer=initializer
                                       , bias_initializer='zeros'
                                       )
        self.proj = tf.keras.layers.Dense(self.out_dim,
                                          kernel_initializer=initializer
                                          , bias_initializer='zeros'
                                          )

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = tf.keras.layers.AveragePooling1D(pool_size=self.N_ratio, strides=self.N_ratio, padding='valid')
            self.norm = tf.keras.layers.BatchNormalization(epsilon=NORM_EPS)

    def call(self, inputs):
        B, N, C = inputs.shape
        q = self.q(inputs)
        q = tf.reshape(q, (tf.shape(q)[0], N, self.num_heads, int(C // self.num_heads)))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        if self.sr_ratio > 1:
            """
                pytorch는 차원이 NCHW, tensorflow 는 NHWC , 즉 채널 차원이 뒤로가있는 텐서플로에 비해 
                파이토치는 앞에 있다. 
                transpose 진행할때 channel 차원에 대한 주의 필요
            """
            # x_ = tf.transpose(inputs, perm=[0, 2, 1])
            x_ = self.sr(inputs)
            x_ = self.norm(x_)
            # x_ = tf.transpose(x_, perm=[0, 2, 1])
            k = self.k(x_)

            k = tf.reshape(k, (tf.shape(q)[0], -1, self.num_heads, int(C // self.num_heads)))
            k = tf.transpose(k, perm=[0, 2, 3, 1])

            v = self.v(x_)
            v = tf.reshape(v, (tf.shape(q)[0], -1, self.num_heads, int(C // self.num_heads)))
            v = tf.transpose(v, perm=[0, 2, 1, 3])
        else:
            k = self.k(inputs)
            k = tf.reshape(k, (tf.shape(q)[0], -1, self.num_heads, int(C // self.num_heads)))
            k = tf.transpose(k, perm=[0, 2, 3, 1])
            v = self.v(inputs)
            v = tf.reshape(v, (tf.shape(q)[0], -1, self.num_heads, int(C // self.num_heads)))
            v = tf.transpose(v, perm=[0, 2, 1, 3])

        attn = tf.matmul(q, k) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.transpose(tf.matmul(attn, v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, (tf.shape(x)[0], N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config


class NTB(tf.keras.layers.Layer):
    """
    Next Transformer Block
    """

    def __init__(self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
                 mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0):
        super(NTB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels=in_channels, out_channels=self.mhsa_out_channels, stride=stride)
        self.norm1 = tf.keras.layers.BatchNormalization(
            epsilon=NORM_EPS
        )
        self.e_mhsa = E_MHSA(dim=self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(in_channels=self.mhsa_out_channels, out_channels=self.mhca_out_channels, stride=1)
        self.mhca = MHCA(out_channels=self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = tf.keras.layers.BatchNormalization(
            epsilon=NORM_EPS
        )
        self.mlp = Mlp(in_features=out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

    def call(self, inputs):
        x = self.patch_embed(inputs)
        B, H, W, C = x.shape
        out = self.norm1(x)
        out = tf.reshape(out, (tf.shape(out)[0], H * W, C))
        e_mhsa_res = self.e_mhsa(out)
        out = self.mhsa_path_dropout(e_mhsa_res)
        x_ = tf.reshape(out, (tf.shape(out)[0], H, W, C))
        x = x + x_
        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))

        """
            채널 위치가 맨 뒤이기 떄문에 concat axis =3
        """
        # x = tf.concat([x, out], 1)
        x = tf.concat([x, out], 3)

        out = self.norm2(x)
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x

    def get_config(self):
        config = super().get_config()
        return config


def NHS(inputs, input_channel, stage_id, path_dropout, depths=None, strides=None, attn_drop=0, head_dim=32,
        mix_block_ratio=0.75,
        drop=0, sr_ratios=None):
    if depths is None:
        depths = [3, 4, 10, 3]
    if sr_ratios is None:
        sr_ratios = [8, 4, 2, 1]
    if strides is None:
        strides = [1, 2, 2, 2]

    stage_out_channels = [[96] * (depths[0]),
                          [192] * (depths[1] - 1) + [256],
                          [384, 384, 384, 384, 512] * (depths[2] // 5),
                          [768] * (depths[3] - 1) + [1024]]

    # Next Hybrid Strategy
    stage_block_types = [[NCB] * depths[0],
                         [NCB] * (depths[1] - 1) + [NTB],
                         [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
                         [NCB] * (depths[3] - 1) + [NTB]]
    dpr = [x.numpy() for x in tf.linspace(0., path_dropout, sum(depths))]
    x = inputs
    idx = 0
    for i in range(stage_id):
        idx += depths[i]
    numrepeat = depths[stage_id]
    output_channels = stage_out_channels[stage_id]
    block_types = stage_block_types[stage_id]
    for block_id in range(numrepeat):
        if strides[stage_id] == 2 and block_id == 0:
            stride = 2
        else:
            stride = 1
        output_channel = output_channels[block_id]
        block_type = block_types[block_id]
        if block_type is NCB:
            x = NCB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                    drop=drop, head_dim=head_dim)(x)
        elif block_type is NTB:
            x = NTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                    sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                    attn_drop=attn_drop, drop=drop)(x)
        input_channel = output_channel
    return x


def NextViT(stem_chs, depths, path_dropout, image_size=(224, 224), attn_drop=0, drop=0, num_classes=1000,
            activation='sigmoid', strides=None, sr_ratios=None, head_dim=32, mix_block_ratio=0.75):
    if sr_ratios is None:
        sr_ratios = [8, 4, 2, 1]
    if strides is None:
        strides = [1, 2, 2, 2]

    input_layer = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 3))
    x = input_layer
    x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode='torch')
    x = ConvBNReLU(stem_chs[0], kernel_size=3, stride=2)(x)
    x = ConvBNReLU(stem_chs[1], kernel_size=3, stride=1)(x)
    x = ConvBNReLU(stem_chs[2], kernel_size=3, stride=1)(x)
    x = ConvBNReLU(stem_chs[2], kernel_size=3, stride=2)(x)

    outputs = []

    for stage_id in range(len(depths)):
        x = NHS(x, stem_chs[-1], stage_id, path_dropout, depths, strides, attn_drop, head_dim, mix_block_ratio, drop,
                sr_ratios)
        outputs.append(x)

    x = tf.keras.layers.BatchNormalization(
        epsilon=NORM_EPS
    )(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(num_classes,
                                        kernel_initializer=initializer,
                                        bias_initializer='zeros',
                                        activation=activation)(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=main_output, name='next_vit')
