# API Contract

Here is a documentation of the field needed by our frontend.

## ex_id
Type : Int
Mandatory : False
ID of the data being processed 

## image
Type : Float matrix

The image processed

## tokens

Type : Array of string

The list of the token sent to the transformer. It must be the text token followed by image token.
Example : ['[CLS]', 'text', 'input', 'for', 'example', str(ex_id), '.', '[SEP]','IMG_0', 'IMG_1', 'IMG_2', 'IMG_3', 'IMG_4', 'IMG_5'].

The token for each image patch are just labels not the actual pixels.

## txt_len

Type : int

Lenght of the texts tokens.

## img_coords
Type : array of tuple 

The (x, y) coordinates of each image token on the original image, assuming the *top left* corner of an image is (0, 0) The order of coordinates should correspond to how image tokens are ordered in 'tokens'

## accuracy

OPTIONAL true if the model predicted the correct label. False otherwise

## attention

Attention weights for all attention heads in all layers 

Shape: (n_layers, n_attention_heads_per_layer, n_tokens, n_tokens)
                
n_layers and n_attention_heads_per_layer should be the same accross example The order of columns and rows of the attention weight matrix for each head should correspond to how tokens are ordered in 'tokens'


## hidden_states

The hidden representations for each token in the model, both before the first layer and after each layer 

Shape: (n_layers + 1, n_tokens, hidden_state_vector_size)

Note that in our demo app, hidden representations of stop words were removed to reduce the number of displayed datapoints

## custom_metrics

OPTIONAL 

Custom statistics for attention heads in all layers

Shape: (n_layers, n_attention_heads_per_layer)

The order should follow how attention heads are ordered in 'attention' matrices