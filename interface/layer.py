
# Get the L1 Distance layer from the DL model

import tensorflow
from tensorflow.keras.layers import Layer

class Distance_layer(Layer): #declaring Distance_layer class which inherits from the Layer clas
    
    # i sat (**kwargs) to accept any parameter
    def __init__(self, **kwargs):
        # i sat it to super to call the Layer class, to ensures that Distance_layer is properly initialized as a Layer
        super().__init__()
       
    # calculate the similarity of two images
    def call(self, input_embedding, validation_embedding):
        return tensorflow.math.abs(input_embedding - validation_embedding)
