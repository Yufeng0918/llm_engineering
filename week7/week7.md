# Week 7 Learning Notes: Fine-tuning an Open-Source Model to Predict Product Prices

This week's focus is on leveraging open-source large language models (LLMs) and efficient fine-tuning techniques, specifically LoRA and QLoRA, to build a system capable of predicting product prices from descriptions. The journey covers model selection, base model evaluation, the fine-tuning process, and final model evaluation.

## Introduction to LoRA and QLoRA

This session introduces the concepts of LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) as efficient methods for fine-tuning large language models. The primary goal is to predict product prices using an open-source model.

#### Colab Pro-Tips for Efficient Workflows

Two crucial pro-tips are highlighted for working with Google Colab, particularly when dealing with `pip` installations and runtime errors:

1.  **Ignoring `pip` compatibility errors:** When running `pip install` commands, you might encounter compatibility errors (e.g., `gcsfs requires fsspec==X, but you have fsspec Y`). These can generally be safely ignored and attempting to fix them by changing version numbers can introduce new problems.
2.  **Addressing "CUDA is required but not available" errors:** This misleading error often indicates that your Colab runtime has been switched due to high demand, not a package version issue. The solution involves:
    *   Disconnecting and deleting the runtime (`Runtime menu >> Disconnect and delete runtime`).
    *   Reloading the Colab notebook and clearing all outputs (`Edit menu >> Clear All Outputs`).
    *   Connecting to a new T4 GPU runtime.
    *   Verifying GPU availability (`View resources` from top-right menu).
    *   Rerunning cells from top to bottom, starting with the `pip` installs.

#### Essential Setup and Libraries

The notebook begins by setting up the environment with necessary Python packages and imports for working with large language models, specifically focusing on `transformers` and `peft` (Parameter-Efficient Fine-Tuning).

**Key Code Snippet: Pip Installs**

```python
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0
!pip install -q datasets requests peft
```

**Key Code Snippet: Imports**

```python
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datetime import datetime
```

#### Model Configuration and HuggingFace Login

Constants are defined for the base model (`meta-llama/Meta-Llama-3.1-8B`) and the fine-tuned model path. Hyperparameters for QLoRA fine-tuning, such as `LORA_R`, `LORA_ALPHA`, and `TARGET_MODULES` (which specify the attention layers to apply LoRA to), are also set.

Access to HuggingFace models requires authentication, which is handled by retrieving a token from Colab's user data secrets.

**Key Code Snippet: Constants**

```python
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
FINETUNED_MODEL = f"ed-donner/pricer-2024-09-13_13.04.39"

# Hyperparameters for QLoRA Fine-Tuning

LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
```

**Key Code Snippet: HuggingFace Login**

```python
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
```

#### Exploring Model Quantization (8-bit and 4-bit)

The session demonstrates loading the base model with different quantization configurations to reduce memory footprint and improve inference speed. This includes loading the model without quantization, then with 8-bit quantization, and finally with 4-bit quantization (NF4, nested quantization, and `bfloat16` compute dtype). Each step emphasizes the significant memory savings achieved with quantization.

A critical point is made about restarting the Colab session between loading different model versions to clear the GPU cache, ensuring accurate memory footprint measurements and preventing conflicts.

**Key Code Snippet: Loading Base Model without Quantization**

```python
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
```

**Key Code Snippet: Loading Base Model using 8-bit Quantization**

```python
quant_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.1f} GB")
```

**Key Code Snippet: Loading Tokenizer and Base Model using 4-bit Quantization**

```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:,.2f} GB")
```

#### Understanding LoRA Adapter Matrices

The session concludes by explaining the structure of LoRA adapter matrices (`lora_A` and `lora_B`) for each target module. These matrices are designed to adapt weights by a small additive change, significantly reducing the number of trainable parameters compared to full fine-tuning. A calculation is provided to estimate the total number of parameters introduced by LoRA.

**Key Code Snippet: LoRA Parameter Calculation**

```python
# Each of the Target Modules has 2 LoRA Adaptor matrices, called lora_A and lora_B
# These are designed so that weights can be adapted by adding alpha * lora_A * lora_B
# Let's count the number of weights using their dimensions:
# See the matrix dimensions above
lora_q_proj = 4096 * 32 + 4096 * 32
lora_k_proj = 4096 * 32 + 1024 * 32
lora_v_proj = 4096 * 32 + 1024 * 32
lora_o_proj = 4096 * 32 + 4096 * 32

# Each layer comes to
lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj
# There are 32 layers
params = lora_layer * 32
# So the total size in MB is
size = (params * 4) / 1_000_000
print(f"Total number of params: {params:,} and size {size:,.1f}MB")
```

## Selecting our model and evaluating the base model against the task

This session focuses on selecting an appropriate base model for the product price prediction task and evaluating its performance before any fine-tuning. A key point emphasized is the significant difference in scale between the 8-billion parameter base model and much larger models like GPT-4o (trillions of parameters), highlighting the challenge and the need for effective fine-tuning.

#### Important Installation Note: Ignoring `fsspec` Errors

A crucial warning is given regarding `pip install` errors related to `fsspec`. Users are instructed to **ignore** these compatibility errors, as the specified `fsspec` version is correct and necessary for HuggingFace to load datasets properly. Attempting to "fix" this error by upgrading `fsspec` can lead to obscure data loading failures later.

**Key Code Snippet: Pip Installs (with warning)**

```python
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q --upgrade requests==2.32.3 bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 datasets==3.2.0 peft==0.14.0 trl==0.14.0 matplotlib
```

#### Imports and Constants

The necessary libraries for data handling, model loading, and evaluation are imported. A set of constants defines potential base models (Llama 3.1, Qwen 2.5, Gemma 2, Phi-3), the HuggingFace dataset name, maximum sequence length, and flags for quantization. Color codes are also defined for visual output during evaluation.

**Key Code Snippet: Imports**

```python
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import matplotlib.pyplot as plt
```

**Key Code Snippet: Tokenizers and Constants**

```python
LLAMA_3_1 = "meta-llama/Meta-Llama-3.1-8B"
QWEN_2_5 = "Qwen/Qwen2.5-7B"
GEMMA_2 = "google/gemma-2-9b"
PHI_3 = "microsoft/Phi-3-medium-4k-instruct"

# Constants

BASE_MODEL = LLAMA_3_1
HF_USER = "ed-donner"
DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182
QUANT_4_BIT = True

# Used for writing to output in color

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

%matplotlib inline
```

#### Tokenizer Investigation and Data Loading

A utility function `investigate_tokenizer` is introduced to explore how different tokenizers (for Llama 3.1, Qwen 2.5, Gemma 2, Phi-3) encode numerical values. This helps in understanding the tokenization behavior of different models. The dataset, previously uploaded to Hugging Face, is then easily loaded using `load_dataset`.

**Key Code Snippet: Tokenizer Investigation Function**

```python
def investigate_tokenizer(model_name):
  print("Investigating tokenizer for", model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  for number in [0, 1, 10, 100, 999, 1000]:
    tokens = tokenizer.encode(str(number), add_special_tokens=False)
    print(f"The tokens for {number}: {tokens}")
```

**Key Code Snippet: Loading Data**

```python
dataset = load_dataset(DATASET_NAME)
train = dataset['train']
test = dataset['test']
```

#### Preparing and Evaluating the Base Model

The chosen base model (Llama 3.1) is loaded with 4-bit quantization, similar to the previous session, to optimize memory usage. A tokenizer is also loaded and configured.

Two critical functions are defined for model interaction and evaluation:
*   `extract_price`: A regex-based function to parse the predicted price from the model's text output.
*   `model_predict`: Takes a prompt, encodes it, generates a response from the base model, and extracts the price using `extract_price`.

Finally, a `Tester` class is implemented to systematically evaluate the model's performance against the test dataset. It calculates absolute errors, Root Mean Squared Logarithmic Error (RMSLE), and classifies prediction accuracy into color-coded categories (green, orange, red) for visualization. The `report` method generates a scatter plot showing ground truth vs. model estimates, providing a clear visual assessment of the base model's current capabilities.

**Key Code Snippet: Quantization Configuration and Model Loading**

```python
if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
  )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:.1f} GB")
```

**Key Code Snippet: Price Extraction and Model Prediction Functions**

```python
def extract_price(s):
    if "Price is $" in s:
      contents = s.split("Price is $")[1]
      contents = contents.replace(',','').replace('$','')
      match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
      return float(match.group()) if match else 0
    return 0

def model_predict(prompt):
    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = base_model.generate(inputs, max_new_tokens=4, attention_mask=attention_mask, num_return_sequences=1)
    response = tokenizer.decode(outputs[0])
    return extract_price(response)
```

**Key Code Snippet: Tester Class for Evaluation**

```python
class Tester:
    def __init__(self, predictor, data, title=None, size=250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []
    def color_for(self, error, truth):
        if error<40 or error/truth < 0.2:
            return "green"
        elif error<80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"
    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint["text"])
        truth = datapoint["price"]
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint["text"].split("\n\n")[1][:20] + "..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")
    def chart(self, title):
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()
    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)
    def run(self):
        self.error = 0
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()
    @classmethod
    def test(cls, function, data):
        cls(function, data).run()
```

## Training!

This session delves into the core process of fine-tuning the selected open-source model using the QLoRA technique. It covers the setup of the training environment, configuration of LoRA and training parameters, and the execution of the fine-tuning process.

#### Important Installation Note Revisited

Similar to the previous sessions, a crucial reminder is given about potential `pip install` errors related to `fsspec`. The instruction remains to **ignore** these errors, as the specified version is correct and necessary for HuggingFace dataset loading.

**Key Code Snippet: Pip Installs**

```python
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q --upgrade requests==2.32.3 bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 datasets==3.2.0 peft==0.14.0 trl==0.14.0 matplotlib wandb
```

#### Imports and Constants for Training

Additional imports for `wandb` (Weights & Biases) and `trl` (Transformer Reinforcement Learning) are added, as these libraries are central to the training and logging process. The constants are expanded to include `PROJECT_NAME`, `HF_USER`, `RUN_NAME` (for unique model saving), and detailed hyperparameters for both QLoRA and the training loop itself (e.g., `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `OPTIMIZER`).

**Key Code Snippet: Imports**

```python
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import matplotlib.pyplot as plt
```

**Key Code Snippet: Constants**

```python
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "ed-donner" # your HF name here!

# Data
DATASET_NAME = f"{HF_USER}/pricer-data"
MAX_SEQUENCE_LENGTH = 182

# Run name for saving the model in the hub
RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Hyperparameters for QLoRA
LORA_R = 32
LORA_ALPHA = 64
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.1
QUANT_4_BIT = True

# Hyperparameters for Training
EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
LR_SCHEDULER_TYPE = 'cosine'
WARMUP_RATIO = 0.03
OPTIMIZER = "paged_adamw_32bit"

# Admin config
STEPS = 50
SAVE_STEPS = 2000
LOG_TO_WANDB = True
```

#### Logging into HuggingFace and Weights & Biases

For tracking experiments and pushing models to the Hugging Face Hub, both platforms require authentication. The notebook details how to log in using tokens stored as secrets in Google Colab. Weights & Biases is configured to log against the defined project name and to save model checkpoints.

**Key Code Snippet: HuggingFace and Weights & Biases Login**

```python
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

wandb_api_key = userdata.get('WANDB_API_KEY')
os.environ["WANDB_API_KEY"] = wandb_api_key
wandb.login()

os.environ["WANDB_PROJECT"] = PROJECT_NAME
os.environ["WANDB_LOG_MODEL"] = "checkpoint" if LOG_TO_WANDB else "end"
os.environ["WANDB_WATCH"] = "gradients"
```

#### Data Loading and Model Preparation

The dataset is loaded, and the base model is prepared with 4-bit quantization, similar to the previous session. The tokenizer is also loaded and configured with a `pad_token` and `padding_side`.

**Key Code Snippet: Model Loading with Quantization**

```python
if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
  )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id
```

#### Data Collator for Completion-Only LM

A critical component for effective fine-tuning in this price prediction task is the `DataCollatorForCompletionOnlyLM` from the `trl` library. This collator ensures that the model is only trained to predict the "completion" part of the sequence (i.e., the price after "Price is $"), ignoring the input prompt. This prevents the model from trying to regenerate the input description, which is not the objective.

**Key Code Snippet: Data Collator Setup**

```python
from trl import DataCollatorForCompletionOnlyLM
response_template = "Price is $"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
```

#### Configuring LoRA and Training Parameters

Two main configuration objects are created:
1.  `LoraConfig`: Defines the LoRA-specific hyperparameters (`lora_alpha`, `lora_dropout`, `r`, `target_modules`), specifying how the low-rank adapters should be applied to the model.
2.  `SFTConfig`: Sets up the overall training parameters, including output directory, number of epochs, batch size, gradient accumulation, learning rate schedule, logging steps, and integration with Weights & Biases and Hugging Face Hub for saving and pushing the fine-tuned model.

**Key Code Snippet: LoRA and SFT Configurations**

```python
lora_parameters = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="no",
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True
)
```

#### Initiating Fine-Tuning with `SFTTrainer`

The `SFTTrainer` class from `trl` orchestrates the fine-tuning process, taking the base model, training dataset, LoRA configuration, training arguments, and the data collator as inputs. The `fine_tuning.train()` method kicks off the training. The notebook also warns about potential Colab runtime interruptions and suggests how to resume training from a saved checkpoint if necessary. Finally, the fine-tuned model is pushed to the Hugging Face Hub for storage and future use.

**Key Code Snippet: `SFTTrainer` Setup and Training**

```python
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    peft_config=lora_parameters,
    args=train_parameters,
    data_collator=collator
  )

fine_tuning.train()

fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")

if LOG_TO_WANDB:
  wandb.finish()
```

## Testing Our Fine-tuned Model

This final session focuses on evaluating the performance of the fine-tuned model against the product price prediction task. It highlights the significant improvements expected from fine-tuning compared to the base model.

#### Installation and Imports

The required libraries for evaluation are installed, including `peft` for loading the fine-tuned model and `matplotlib` for visualization. Imports are similar to previous sessions, tailored for model loading and evaluation.

**Key Code Snippet: Pip Installs**

```python
!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
!pip install -q --upgrade requests==2.32.3 bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 datasets==3.2.0 peft==0.14.0 trl==0.14.0 matplotlib wandb
```

**Key Code Snippet: Imports**

```python
import os
import re
import math
from tqdm import tqdm
from google.colab import userdata
from huggingface_hub import login
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
from peft import PeftModel
import matplotlib.pyplot as plt
```

#### Constants and HuggingFace Login

Constants are defined for the base model, project name, HuggingFace user, and crucially, the `RUN_NAME` and `REVISION` of the specific fine-tuned model to be loaded from the Hugging Face Hub. This allows for precise loading of the trained model checkpoint. HuggingFace login is performed to access the model.

**Key Code Snippet: Constants**

```python
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "ed-donner" # your HF name here! Or use mine if you just want to reproduce my results.

# The run itself
RUN_NAME = "2024-09-13_13.04.39"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "e8d637df551603dc86cd7a1598a8f44af4d7ae36" # or REVISION = None
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Uncomment this line if you wish to use my model
# FINETUNED_MODEL = f"ed-donner/{PROJECT_RUN_NAME}"

# Data
DATASET_NAME = f"{HF_USER}/pricer-data"
# Or just use the one I've uploaded
# DATASET_NAME = "ed-donner/pricer-data"

# Hyperparameters for QLoRA
QUANT_4_BIT = True

%matplotlib inline

# Used for writing to output in color
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}
```

#### Loading the Fine-tuned Model

The base model is first loaded with 4-bit quantization, and then the `PeftModel.from_pretrained` method is used to load the LoRA adapters on top of the base model. This efficiently combines the base model with the fine-tuned weights, preparing it for inference.

**Key Code Snippet: Loading Fine-tuned Model**

```python
if QUANT_4_BIT:
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
  )
else:
  quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
  )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
base_model.generation_config.pad_token_id = tokenizer.pad_token_id

# Load the fine-tuned model with PEFT
if REVISION:
  fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)
else:
  fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)

print(f"Memory footprint: {fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")
```

#### Price Extraction and Improved Prediction Function

The `extract_price` function from the previous sessions is reused to parse the numerical price from the model's text output.

A significant improvement is introduced in the `improved_model_predict` function. Instead of just taking the most likely next token, this function considers the weighted average of the top-K predicted tokens. This is particularly effective when the model might predict slightly different numerical values for the price, and taking an average can lead to more robust and accurate predictions. The code takes advantage of the fact that Llama models can generate a 3-digit number as a single token.

**Key Code Snippet: Improved Model Prediction Function**

```python
top_K = 3

def improved_model_predict(prompt, device="cuda"):
    set_seed(42)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, device=device)

    with torch.no_grad():
        outputs = fine_tuned_model(inputs, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :].to('cpu')

    next_token_probs = F.softmax(next_token_logits, dim=-1)
    top_prob, top_token_id = next_token_probs.topk(top_K)
    prices, weights = [], []
    for i in range(top_K):
      predicted_token = tokenizer.decode(top_token_id[0][i])
      probability = top_prob[0][i]
      try:
        result = float(predicted_token)
      except ValueError as e:
        result = 0.0
      if result > 0:
        prices.append(result)
        weights.append(probability)
    if not prices:
      return 0.0, 0.0
    total = sum(weights)
    weighted_prices = [price * weight / total for price, weight in zip(prices, weights)]
    return sum(weighted_prices).item()
```

#### Final Evaluation

The `Tester` class (from the previous sessions) is reused to run the evaluation of the fine-tuned model using the `improved_model_predict` function. The results are presented with average error, RMSLE, and hit rate, along with a scatter plot for visual analysis. The notebook concludes by comparing the fine-tuned model's performance to the base Llama 3.1 model and even GPT-4o, emphasizing the effectiveness of the fine-tuning process. A caveat about predicting sale prices is also included, reminding the user that the model's predictions are based on the data it was trained on.
