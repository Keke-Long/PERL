import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.constraints import Constraint


class ScalarMinMaxConstraint(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}


# 1. 自定义IDM模型的层 #arg = (22.165, 0.922, 2.823, 1.546, 1.225) # IDN_US101第二次标定
# class IDM_Layer(tf.keras.layers.Layer):
#     def __init__(self, forward_steps, **kwargs):
#         self.forward_steps = forward_steps
#         super(IDM_Layer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # 初始化IDM参数
#         self.vf = self.add_weight(name='vf',
#                                   shape=(),
#                                   initializer=Constant(22.165),
#                                   trainable=True,
#                                   constraint=ScalarMinMaxConstraint(0., 30.))
#
#         self.A = self.add_weight(name='A',
#                                  shape=(),
#                                  initializer=Constant(0.922),
#                                  trainable=True,
#                                  constraint=ScalarMinMaxConstraint(0., 5.))
#
#         self.b = self.add_weight(name='b',
#                                  shape=(),
#                                  initializer=Constant(2.823),
#                                  trainable=True,
#                                  constraint=ScalarMinMaxConstraint(0., 5.))
#
#         self.s0 = self.add_weight(name='s0',
#                                   shape=(),
#                                   initializer=Constant(1.546),
#                                   trainable=True,
#                                   constraint=ScalarMinMaxConstraint(0., 10.))
#
#         self.T = self.add_weight(name='T',
#                                  shape=(),
#                                  initializer=Constant(1.225),
#                                  trainable=True,
#                                  constraint=ScalarMinMaxConstraint(1., 5.))
#
#         super(IDM_Layer, self).build(input_shape)
#
#     def call(self, inputs):
#         vi, delta_v, delta_d = inputs
#         predictions = []
#
#         for _ in range(self.forward_steps):
#             s_star = self.s0 + tf.maximum(tf.constant(0.0, dtype=tf.float32), vi*self.T + (vi * delta_v) / (2 * (self.A*self.b) ** 0.5))
#             epsilon = 1e-5
#             ahat = self.A * (1 - (vi/self.vf)**4 - (s_star / (delta_d + epsilon)) ** 2)
#             predictions.append(ahat)
#
#             # 更新为下一个时间步的输入
#             vi = vi + ahat*0.1
#             delta_v = delta_v - ahat*0.1
#
#         return tf.stack(predictions, axis=1)
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "forward_steps": self.forward_steps
#         })
#         return config



class Newell_Layer(tf.keras.layers.Layer):
    def __init__(self, forward_steps, **kwargs):
        self.forward_steps = forward_steps
        super(Newell_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化可训练的参数 w
        self.w = self.add_weight(name='w',
                                 shape=(),
                                 initializer=Constant(4.1),
                                 trainable=True,
                                 constraint=ScalarMinMaxConstraint(0., 10.0))  # 根据需要调整约束

        super(Newell_Layer, self).build(input_shape)

    def call(self, inputs):
        vi, delta_y, v_previous, x_input = inputs
        predictions = []

        def get_ahat(i, x_input):
            def body(j, found_valid_data, ahat):
                d = tf.reduce_sum(x_input[:, -1, :j], axis=-1)
                time_difference = d * 150 / (self.w + x_input[:, -1, 4 + j] * 25)
                referred_index = tf.cast(i - time_difference * 10, tf.int32)
                referred_index = tf.clip_by_value(referred_index, 0, tf.shape(x_input)[1] - 1)
                ahat_temp = tf.gather(x_input, referred_index, axis=1, batch_dims=1)[:, 9 + j]

                # 更新标志和 ahat
                found_valid_data_new = tf.logical_or(found_valid_data, tf.reduce_any(ahat_temp != 0))
                ahat_new = tf.cond(found_valid_data_new, lambda: ahat_temp, lambda: ahat)

                return j + 1, found_valid_data_new, ahat_new

            def condition(j, found_valid_data, _):
                return tf.logical_and(j < 5, tf.logical_not(found_valid_data))

            # 初始化循环变量为张量
            j = tf.constant(1, dtype=tf.int32)
            found_valid_data = tf.constant(False)
            ahat = tf.zeros([tf.shape(x_input)[0]], dtype=tf.float32)  # 初始化为向量

            # shape_invariants 定义了循环变量的形状不变量
            shape_invariants = [j.get_shape(), found_valid_data.get_shape(), tf.TensorShape([None])]

            # 循环直到找到有效数据或检查了所有车辆
            _, _, ahat = tf.while_loop(condition, body, [j, found_valid_data, ahat], shape_invariants=shape_invariants)

            return ahat

        # 使用可训练参数 w 计算 Newell 模型
        for i in range(self.forward_steps):
            ahat = get_ahat(i, x_input)
            predictions.append(ahat)

        return tf.stack(predictions, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "forward_steps": self.forward_steps
        })
        return config
