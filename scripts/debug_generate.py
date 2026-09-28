import sys
import os
import torch
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from configs.yabgu_50m import YabguConfig
from models.transformer import YabguTransformer

device = "cpu"
tokenizer = Tokenizer.from_file("tokenizer/vocab/tokenizer.json")
config = YabguConfig()
model = YabguTransformer(config)

ckpt = torch.load("checkpoints/yabgu_50m_instruct_fp16.pt", map_location=device, weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt)
state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Encode WITHOUT post-processor adding <|eos|> at the end of the prompt!
tokenizer.no_padding()
tokenizer.post_processor = None

prompt_str = "<|bos|>### Сұрақ:\nСен кімсің?\n\n### Жауап:\n"
tokens = tokenizer.encode(prompt_str).ids
print(f"Clean prompt tokens: {tokens}")
print(f"Clean prompt decoded: {repr(tokenizer.decode(tokens))}")

t = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    logits, _ = model(t)
    last_logits = logits[0, -1]
    top_probs, top_indices = torch.topk(torch.softmax(last_logits, dim=-1), 10)
    print("\nTop 10 predicted tokens after CLEAN prompt:")
    for prob, idx in zip(top_probs, top_indices):
        idx_item = idx.item()
        token_str = tokenizer.decode([idx_item])
        print(f"  Token {idx_item} ('{token_str}'): prob {prob.item():.4f}")
