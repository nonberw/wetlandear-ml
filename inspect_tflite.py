import collections, sys
from ai_edge_litert.interpreter import Interpreter
M = r'C:\Users\Timur\Documents\wetlandear-app\models'
for name in ('birdnet_v2.4_fp16.tflite', 'birdnet_v2.4_meta_fp16.tflite', 'yamnet.tflite'):
    it = Interpreter(model_path=M + '\\' + name, experimental_preserve_all_tensors=True); it.allocate_tensors()
    ops = collections.Counter(o['op_name'] for o in it._get_ops_details())
    i, o = it.get_input_details(), it.get_output_details()
    print(name); print('  inputs ', [(d['name'][:30], d['shape'].tolist(), str(d['dtype']).split('.')[-1][:-2]) for d in i])
    print('  outputs', [(d['name'][:30], d['index'], d['shape'].tolist()) for d in o])
    print('  ops    ', dict(ops))
