import itertools
import subprocess

# Define hyperparameter ranges


patch_window_combinations = [
    (4, 7),   # Patch size 4 only with window size 7
    (28, 2)   # Patch size 28 only with window size 2
]
mlp_sizes = [96, 192]
num_transformer_layers = [ "2, 2, 2", "1, 1, 1"]
num_attention_heads = [ "2 ,2 ,2", "4 ,4 ,4"]
embedding_dims = [48, 96]
learning_rates = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
optimizers = ["Adam", "SGD"]
batch_sizes = [16, 64] # 64 instead of 128, training dataset contains 100 samples and with batch_size 128 returns none

# Generate all combinations
param_combinations = list(itertools.product(
    patch_window_combinations,
    mlp_sizes,
    num_transformer_layers,
    num_attention_heads,
    embedding_dims,
    learning_rates,
    optimizers,
    batch_sizes
))

# Fixed parameters
positional_encoding = "learned"
scheduler_used = "" # False
attn_dropout = 0.0
mlp_dropout = 0.0
num_epochs = 500
num_workers = 1

# Neptune credentials (update if necessary)
neptune_project = "GRAINS/visual-sudoku"
neptune_api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly\
9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMDQ2YThmNS1jNWU0LTQxZDItYTQxNy1lMGYzNTM4MmY5YTgifQ=="

directory = "./"

# Loop over each combination
for params in param_combinations:

    patch_window_combination, mlp_size, num_transformer_layer, num_attention_head, embedding_dim, learning_rate, optimizer, batch_size = params

    vit_mlp_ratio = mlp_size / embedding_dim
    patch_size, window_size = patch_window_combination

    # Construct command
    command = f"""
    python ./main.py \
       --patch_size {patch_size} \
       --window_size {window_size} \
       --image_size 112 \
       --in_channels 1 \
       --embed_dim {embedding_dim} \
       --num_layers {num_transformer_layer} \
       --num_heads {num_attention_head} \
       --vit_mlp_ratio {vit_mlp_ratio} \
       --weight_decay 0.1 \
       --batch_size {batch_size} \
       --lr {learning_rate} \
       --epochs {num_epochs} \
       --warmup_epochs 10 \
       --mlp_dropout {mlp_dropout} \
       --attn_dropout {attn_dropout} \
       --scheduler "{scheduler_used} \
       --min_lr 1e-6 \
       --clip_grad 3.0 \
       --neptune_project "{neptune_project}" \
       --neptune_api_token "{neptune_api_token}" \
       --dir "{directory}"
       --num_workers "{num_workers}" 
    """

    # Execute the command
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())

    # Print errors (if any)
    for line in process.stderr:
        print("ERROR:", line.strip())

    # Wait for the process to complete
    process.wait()
    print("Process completed with exit code:", process.returncode)
    # print("Executing:", command)