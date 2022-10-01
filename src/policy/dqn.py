from telnetlib import IP
from src.policy.policy import Policy
import tensorflow as tf


def make_q_network(state_shape, action_size, learning_rate) -> tf.keras.models.Model:
    """Makes a simple q network for experimentation"""
    state_input = tf.keras.layers.Input(state_shape, name="states")
    actions_input = tf.keras.layers.Input((action_size,), name="actions_mask")
    
    hidden_1 = tf.keras.layers.Dense(10, activation="relu")(state_input)
    hidden_2 = tf.keras.layers.Dense(10, activation="relu")(hidden_1)
    q_values = tf.keras.layers.Dense(action_size)(hidden_2)

    masked_q_values = tf.keras.layers.Multiply()([q_values,actions_input])

    model = tf.keras.models.Model(inputs=[state_input,actions_input], outputs=masked_q_values)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(loss="mse", optimizer=optimizer)
    return model

class DQNPolicy(Policy):
    
    def __init__(self, state_shape: int, action_size: int) -> None:
        super().__init__()
        self.state_shape = state_shape
        self.action_size = action_size

        self.q_network = make_q_network(self.state_shape, self.action_size, 0.01)
        self.q_network.predict()

        
        
        
    def get_greedy_action(self, state) -> int:
        return super().get_greedy_action(state)
