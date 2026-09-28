import sys
import os

# Clean module cache to ensure fresh code
for mod in list(sys.modules.keys()):
    if mod.startswith(("models", "training", "configs")):
        del sys.modules[mod]

sys.path.insert(0, "/content")

from training.sft_trainer import train_sft

if __name__ == "__main__":
    base_ckpt = "/content/checkpoints/checkpoint_step_3000.pt"
    if not os.path.exists(base_ckpt):
        base_ckpt = "/content/checkpoints/yabgu_50m_scale3000_fp16.pt"
        
    data_path = "/content/data/sft/kazakh_sft_combined.jsonl"
    tokenizer_path = "/content/tokenizer/vocab/tokenizer.json"
    output_dir = "/content/checkpoints"
    
    print("=================================================================")
    print(" STARTING YABGU-50M SFT (INSTRUCTION TUNING) ON TESLA T4 GPU")
    print("=================================================================")
    print(f"Base checkpoint: {base_ckpt}")
    print(f"Data path: {data_path}")
    print(f"Tokenizer: {tokenizer_path}")
    
    final_fp16 = train_sft(
        base_checkpoint=base_ckpt,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        epochs=1,
        max_steps=1000,
        batch_size=8,
        grad_accum_steps=2,
        learning_rate=1.2e-4,
        min_lr=1.0e-5,
        warmup_steps=50,
        max_seq_len=512,
        eval_interval=50,
        save_interval=100
    )
    print(f"SFT Training Finished! Checkpoint saved at: {final_fp16}")
