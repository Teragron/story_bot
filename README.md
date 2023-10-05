# story_bot
A Telegram bot that uses karpathy's TinyStories42M Model to generate stories

The App is easy to deploy on pythonanywhere

out_dir = "out"
eval_interval = 10
log_interval = 1
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "llamac"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 512
vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
# model
dim = 192
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.1
# adamw optimizer
gradient_accumulation_steps = 8  # used to simulate larger batch sizes
learning_rate = 5e-4  # max learning rate
max_iters = 2400  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 1000  # how many steps to warm up for





