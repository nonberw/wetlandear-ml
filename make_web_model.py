"""Give BirdNET a second output: the 1024-number sound fingerprint (the tensor feeding the last layer). The server reads it with
experimental_preserve_all_tensors; a browser runtime can only read model outputs, so we list that tensor as an output.
The file is patched in place at byte level: a new 2-element outputs vector is appended and the subgraph points to it."""
import struct, sys
import numpy as np
import tflite
from ai_edge_litert.interpreter import Interpreter
M = r'C:\Users\Timur\Documents\wetlandear-app\models'
SRC = M + r'\birdnet_v2.4_fp16.tflite'
DST = M + r'\birdnet_v2.4_fp16_emb.tflite'
buf = bytearray(open(SRC, 'rb').read())
m = tflite.Model.GetRootAsModel(buf, 0)
sg = m.Subgraphs(0)
outs = [sg.Outputs(i) for i in range(sg.OutputsLength())]
assert len(outs) == 1, outs
emb = outs[0] - 1
field_pos = sg._tab.Pos + sg._tab.Offset(8)
while len(buf) % 4: buf.append(0)
new_pos = len(buf)
buf += struct.pack('<I', 2) + struct.pack('<ii', outs[0], emb)
struct.pack_into('<I', buf, field_pos, new_pos - field_pos)
open(DST, 'wb').write(buf)
print('written', DST, len(buf) // 1024, 'KB; outputs', outs[0], emb)

# check: the patched model gives the same logits and the same fingerprint as the original
a = Interpreter(model_path=SRC, experimental_preserve_all_tensors=True); a.allocate_tensors()
b = Interpreter(model_path=DST); b.allocate_tensors()
print('outputs of patched model:', [(d['index'], d['shape'].tolist()) for d in b.get_output_details()])
x = (np.random.default_rng(0).standard_normal((1, 144000)) * 0.1).astype('float32')
a.set_tensor(a.get_input_details()[0]['index'], x); a.invoke()
b.set_tensor(b.get_input_details()[0]['index'], x); b.invoke()
la = a.get_tensor(a.get_output_details()[0]['index'])[0]; ea = a.get_tensor(a.get_output_details()[0]['index'] - 1)[0]
od = b.get_output_details(); lb = b.get_tensor(od[0]['index'])[0]; eb = b.get_tensor(od[1]['index'])[0]
print('logits max diff', float(np.abs(la - lb).max()), '| fingerprint max diff', float(np.abs(ea - eb).max()), '| fingerprint shape', eb.shape)
