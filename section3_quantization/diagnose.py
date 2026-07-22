import sys, traceback
sys.path.insert(0, '.')
import transformers, torch
print('transformers:', transformers.__version__)
print('torch:', torch.__version__)

from src.engines.transformers_engine import HFTransformersEngine
eng = HFTransformersEngine(model_id='Qwen/Qwen2.5-3B-Instruct', precision='fp16', device_mode='auto')
eng.load()
print('MODEL DEVICE:', eng._model.device)

messages = [{'role': 'user', 'content': 'hello'}]
try:
    ids = eng._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt')
    print('TEMPLATE OK, TYPE:', type(ids))
    ids = ids.to(eng._model.device)
    print('MOVED TO DEVICE OK, SHAPE:', ids.shape)
    out = eng._model.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=eng._tokenizer.eos_token_id)
    print('GENERATE OK:', out.shape)
except Exception:
    print('--- FULL TRACEBACK ---')
    traceback.print_exc()