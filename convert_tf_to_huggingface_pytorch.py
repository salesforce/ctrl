import tensorflow as tf
import tqdm
import torch
import os
import argparse
import sys

from transformers import CTRLConfig
from transformers import CTRLLMHeadModel, CTRLTokenizer
from tensorflow.python import pywrap_tensorflow

parser = argparse.ArgumentParser(description='Code for converting TF checkpoint to PyTorch')
parser.add_argument('--tf_checkpoint', type=str, required=True,
                    help='location of the .data file of the TensorFlow checkpoint. This is NOT the model folder. This could be <path>/seqlen256_v1.ckpt/model.ckpt-413000.data-00000-of-00001')
parser.add_argument('--pytorch_checkpoint', type=str, default='pytorch_model.bin',
                    help='location of where to write the PyTorch checkpoint')
parser.add_argument('--num_layers', type=int, default=48,
                    help='number of layers in the model being converted')

args = parser.parse_args()

model = CTRLLMHeadModel(CTRLConfig())

if os.path.isfile(args.tf_checkpoint):
    print('INFO :: Found TensorFlow checkpoint')
else:
    print('INFO :: TensorFlow checkpoint not found. Please verify location of the .data file or raise GitHub issue if problem persists.')
    
if os.path.isfile(args.pytorch_checkpoint):
    print('PyTorch model already exists. Will not over-write. Please delete old checkpoint or specify different file name')
    sys.exit(1)


chkpt_for_reader = '.'.join(args.tf_checkpoint.split('.')[:-1])
reader = pywrap_tensorflow.NewCheckpointReader(chkpt_for_reader)

tensor_read_get = lambda x, y: torch.tensor(reader.get_tensor(x))
def tensor_read_get(varname, transpose=True):
    loaded_weight = torch.tensor(reader.get_tensor(varname))
    if transpose and len(loaded_weight.shape)>1:
        return loaded_weight.t()
    else:
        return loaded_weight
model.transformer.w.weight.data = tensor_read_get('w', transpose=False)
model.lm_head.bias.data = tensor_read_get('b')
model.transformer.layernorm.weight.data = tensor_read_get('encoder/layer_normalization_96/gamma')
model.transformer.layernorm.bias.data = tensor_read_get('encoder/layer_normalization_96/beta')

list_of_variables = list(filter(lambda x: 'Adagrad' not in x, reader.get_variable_to_shape_map().keys()))

if args.num_layers != 48:
    raise NotImplementedError('Only supports 48 layers at the moment')

for i in tqdm.tqdm(range(args.num_layers)):
    if i==0:
        layer_variables = sorted(filter(lambda x: 'layer/' in x, list_of_variables))
    else:
        layer_variables = sorted(filter(lambda x: 'layer_'+str(i)+'/' in x, list_of_variables))

    current_layer = model.transformer.h[i]

    current_layer.layernorm1.bias.data =  tensor_read_get(layer_variables[0])
    current_layer.layernorm1.weight.data =  tensor_read_get(layer_variables[1])

    current_layer.layernorm2.bias.data =  tensor_read_get(layer_variables[2])
    current_layer.layernorm2.weight.data =  tensor_read_get(layer_variables[3])


    current_layer.multi_head_attention.Wq.bias.data =  tensor_read_get(layer_variables[4])
    current_layer.multi_head_attention.Wq.weight.data =  tensor_read_get(layer_variables[5])
    current_layer.multi_head_attention.Wk.bias.data =  tensor_read_get(layer_variables[6])
    current_layer.multi_head_attention.Wk.weight.data =  tensor_read_get(layer_variables[7])
    current_layer.multi_head_attention.Wv.bias.data =  tensor_read_get(layer_variables[8])
    current_layer.multi_head_attention.Wv.weight.data =  tensor_read_get(layer_variables[9])
    current_layer.multi_head_attention.dense.bias.data =  tensor_read_get(layer_variables[10])
    current_layer.multi_head_attention.dense.weight.data =  tensor_read_get(layer_variables[11])
    current_layer.ffn[0].bias.data =  tensor_read_get(layer_variables[12])
    current_layer.ffn[0].weight.data =  tensor_read_get(layer_variables[13])
    current_layer.ffn[2].bias.data =  tensor_read_get(layer_variables[14])
    current_layer.ffn[2].weight.data =  tensor_read_get(layer_variables[15])

torch.save(model.state_dict(), args.pytorch_checkpoint)
print('INFO :: Saved PyTorch model to ', args.pytorch_checkpoint)
