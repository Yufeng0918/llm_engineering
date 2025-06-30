# Week 6: Building a Product Pricer - From Data Curation to Fine-Tuning

This week focuses on building a "Product Pricer" model capable of estimating the cost of items based on their descriptions. The journey covers extensive data curation, traditional machine learning baselines, testing with frontier models, and finally, fine-tuning a small language model.

## Data Curation Part 1 - Home Appliances and Item Class

The initial step involves data curation, starting with a subset of the Amazon Reviews 2023 dataset: Home Appliances. The goal is to clean and prepare this data for model training.

**Key Concepts:**
-   **Dataset Loading**: Utilizing HuggingFace's `load_dataset` to import product data.
-   **Data Exploration**: Analyzing the loaded data, including titles, descriptions, features, details, and prices.
-   **Price Distribution Analysis**: Plotting histograms to understand the distribution of prices and text lengths to identify outliers and common ranges.
-   **Item Class**: Introduction of a custom `Item` class (`items.py`) to encapsulate a cleaned and curated product-price pair. This class handles:
    -   **Text Scrubbing**: Removing irrelevant characters and numerical product codes from descriptions.
    -   **Tokenization**: Truncating text content to a specific token count (e.g., 160 tokens) using a pre-trained tokenizer (Llama 3.1 8B's tokenizer). This is identified as a "hyper-parameter" determined by trial and error, balancing information content with training efficiency.
    -   **Prompt Generation**: Formatting the product description into a prompt suitable for training and testing language models.

**Important Code Snippets:**

**Loading Dataset:**
```python
from datasets import load_dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split="full", trust_remote_code=True)
```

**Item Class Initialization (from `items.py`):**
```python
from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

MIN_TOKENS = 150 
MAX_TOKENS = 160 
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)

    def make_prompt(self, text):
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX
```

## Data Curation Part 2 - Scaling Up and Balancing the Dataset

This session extends the data curation to include a broader range of product categories, aiming for a massive and balanced dataset for training.

**Key Concepts:**
-   **Scaling Up**: Combining multiple product datasets (e.g., Automotive, Electronics, Office Products) to create a large-scale dataset of 400,000 items.
-   **ItemLoader**: Introduction of the `ItemLoader` class (`loaders.py`) to efficiently load and process large datasets in parallel using `ProcessPoolExecutor`. This significantly speeds up the data loading and scrubbing process.
-   **Dataset Balancing**: Addressing data imbalance, particularly skew towards cheaper items and certain categories (like Automotive). The process involves:
    -   Creating price "slots" to group items by their rounded price.
    -   Sampling from these slots, giving more weight to items from under-represented categories to achieve a more even distribution of prices and categories.
-   **Train/Test Split**: Dividing the curated data into training (400,000 items) and testing (2,000 items) sets.
-   **Data Persistence**: Saving the processed training and testing datasets as pickle files (`train.pkl`, `test.pkl`) to avoid re-running the extensive data curation process in future sessions.

**Important Code Snippets:**

**ItemLoader (from `loaders.py`):**
```python
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from items import Item

CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49

class ItemLoader:
    def __init__(self, name):
        self.name = name
        self.dataset = None

    def load_in_parallel(self, workers):
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        for result in results:
            result.category = self.name
        return results
            
    def load(self, workers=8):
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.name}", split="full", trust_remote_code=True)
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins", flush=True)
        return results
```

**Dataset Balancing Logic:**
```python
import numpy as np
import random
from collections import defaultdict, Counter

# ... (items loading from multiple datasets) ...

slots = defaultdict(list)
for item in items:
    slots[round(item.price)].append(item)

np.random.seed(42)
random.seed(42)
sample = []
for i in range(1, 1000):
    slot = slots[i]
    if i>=240:
        sample.extend(slot)
    elif len(slot) <= 1200:
        sample.extend(slot)
    else:
        weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
        weights = weights / np.sum(weights)
        selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
        selected = [slot[i] for i in selected_indices]
        sample.extend(selected)

random.shuffle(sample)
train = sample[:400_000]
test = sample[400_000:402_000]
```

## Baseline Models - Traditional Machine Learning Approaches

This day focuses on establishing baseline models using traditional machine learning techniques to predict product prices. These models serve as a benchmark to evaluate the performance of more advanced models later.

**Key Concepts:**
-   **Test Harness (`Tester` class)**: A robust utility (`testing.py`) designed to evaluate any prediction function against a subset of the test data (250 items). It visualizes results, calculates average error, Root Mean Squared Log Error (RMSLE), and "hits" (predictions within a certain error margin).
-   **Baseline Models**:
    -   **Random Pricer**: A simple model that returns a random price within the expected range, demonstrating the lowest possible performance.
    -   **Constant Pricer**: A slightly more sophisticated baseline that always returns the average price from the training data.
-   **Feature Engineering**: Extracting structured features from unstructured text product details, such as "Item Weight," "Best Sellers Rank," text length, and whether the brand is a "Top Electronics Brand."
-   **Traditional ML Models**:
    -   **Linear Regression**: Applied to engineered features.
    -   **Bag of Words (BoW) with Linear Regression**: Using `CountVectorizer` to transform text into numerical features, then applying linear regression.
    -   **Word2Vec with Linear Regression**: Using `gensim.models.Word2Vec` to create word embeddings and averaging them to represent document vectors, followed by linear regression.
    -   **Support Vector Regression (SVR)**: A more advanced regression model applied to Word2Vec embeddings.
    -   **Random Forest Regressor**: An ensemble learning method applied to Word2Vec embeddings, generally providing higher accuracy.

**Important Code Snippets:**

**Tester Class (from `testing.py`):**
```python
import math
import matplotlib.pyplot as plt

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

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
        # ... (logic to run prediction and print colored output) ...
        pass

    def chart(self, title):
        # ... (logic to plot scatter chart of ground truth vs. model estimate) ...
        pass

    def report(self):
        # ... (logic to calculate and print average error, RMSLE, and hits) ...
        pass

    def run(self):
        # ... (main execution loop for testing) ...
        pass

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()
```

**Example: Linear Regression Pricer:**
```python
import pandas as pd
from sklearn.linear_model import LinearRegression

# ... (feature extraction functions like get_weight_with_default, get_rank_with_default, get_text_length, is_top_electronics_brand) ...

def get_features(item):
    return {
        "weight": get_weight_with_default(item),
        "rank": get_rank_with_default(item),
        "text_length": get_text_length(item),
        "is_top_electronics_brand": 1 if is_top_electronics_brand(item) else 0
    }

# Assuming train and test are loaded from pickle files
train_df = pd.DataFrame([get_features(item) for item in train])
train_df['price'] = [item.price for item in train]

model = LinearRegression()
model.fit(train_df[['weight', 'rank', 'text_length', 'is_top_electronics_brand']], train_df['price'])

def linear_regression_pricer(item):
    features = get_features(item)
    features_df = pd.DataFrame([features])
    return model.predict(features_df)[0]

# Tester.test(linear_regression_pricer, test)
```

## Enter The Frontier! - Testing with Large Language Models

This session explores the performance of powerful pre-trained Large Language Models (LLMs), referred to as "Frontier Models," in estimating product prices. Unlike the traditional ML models, these LLMs are *not* explicitly trained on our custom dataset; they are evaluated directly on the test set.

**Key Concepts:**
-   **Zero-Shot/Few-Shot Evaluation**: Frontier models are used in a zero-shot (or few-shot, if the prompt includes examples) manner, leveraging their vast pre-training knowledge to understand product descriptions and infer prices.
-   **Test Contamination**: Acknowledging the possibility that these large models might have already seen the product data during their extensive pre-training, potentially giving them an unfair advantage.
-   **Human Baseline**: A "human pricer" is introduced, which reads product descriptions from `human_input.csv` and provides manual price estimates (from `human_output.csv`). This serves as another crucial baseline for comparison.
-   **Prompt Engineering for LLMs**: Crafting a concise system message and user prompt for LLMs to encourage them to output only the price. This involves removing unnecessary phrasing from the original `test_prompt`.
-   **Price Extraction**: A utility function (`get_price`) is developed to parse the LLM's text response and extract the numerical price.
-   **LLM Evaluation**: Testing various OpenAI (GPT-4o-mini, GPT-4o) and Anthropic (Claude 3.5 Sonnet) models.

**Important Code Snippets:**

**Human Pricer Setup:**
```python
import csv

# Assuming test is loaded from pickle file
with open('human_input.csv', 'w', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for t in test[:250]:
        writer.writerow([t.test_prompt(), 0]) # 0 is a placeholder for human input

# Manual intervention: human would fill human_output.csv based on human_input.csv

human_predictions = []
with open('human_output.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        human_predictions.append(float(row[1]))

def human_pricer(item):
    idx = test.index(item)
    return human_predictions[idx]

# Tester.test(human_pricer, test)
```

**Prompt and Price Extraction for LLMs:**
```python
import re

def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": "Price is $"} # Assistant's starting phrase to guide output
    ]

def get_price(s):
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else 0
```

**Example: GPT-4o-mini Pricer:**
```python
from openai import OpenAI
openai = OpenAI()

def gpt_4o_mini(item):
    response = openai.chat.completions.create(
        model="gpt-4o-mini", 
        messages=messages_for(item),
        seed=42,
        max_tokens=5 # Limit response to just the price
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# Tester.test(gpt_4o_mini, test)
```

## Fine-Tuning - Customizing a Small LLM for Price Estimation

The final and most anticipated step is fine-tuning a small language model, specifically `gpt-4o-mini`, using a subset of the curated dataset. This allows the model to specialize in the task of price estimation.

**Key Concepts:**
-   **Fine-Tuning Data Preparation**: Converting a portion of the training data into JSONL (JSON Lines) format, which is required by OpenAI's fine-tuning API. Each entry includes the system message, user prompt, and the desired assistant response (the accurate price).
-   **Training and Validation Splits**: Creating small fine-tuning training (e.g., 200-500 examples) and validation (e.g., 50 examples) sets from the main training dataset. OpenAI recommends relatively small populations for fine-tuning.
-   **OpenAI Fine-tuning API**: Utilizing the OpenAI API to:
    -   Upload the prepared JSONL files for training and validation.
    -   Create a fine-tuning job, specifying the base model (`gpt-4o-mini`), hyperparameters (e.g., `n_epochs=1`), and optional integrations like Weights & Biases for monitoring.
-   **Model Testing**: Once the fine-tuning job is complete and a `fine_tuned_model_name` is available, the newly trained model is tested using the `Tester` class on the unseen test dataset.

**Important Code Snippets:**

**Messages for Fine-tuning (including assistant's target response):**
```python
def messages_for(item):
    system_message = "You estimate prices of items. Reply only with the price, no explanation"
    user_prompt = item.test_prompt().replace(" to the nearest dollar","").replace("\n\nPrice is $","")
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": f"Price is ${item.price:.2f}"} # Key difference: includes the true price
    ]
```

**JSONL Conversion and File Writing:**
```python
import json

def make_jsonl(items):
    result = ""
    for item in items:
        messages = messages_for(item)
        messages_str = json.dumps(messages)
        result += '{"messages": ' + messages_str +'}\n'
    return result.strip()

def write_jsonl(items, filename):
    with open(filename, "w") as f:
        jsonl = make_jsonl(items)
        f.write(jsonl)

# Example usage:
# write_jsonl(fine_tune_train, "fine_tune_train.jsonl")
# write_jsonl(fine_tune_validation, "fine_tune_validation.jsonl")
```

**OpenAI Fine-tuning Job Creation:**
```python
# Assuming train_file and validation_file are uploaded OpenAI File objects
from openai import OpenAI
openai = OpenAI()

wandb_integration = {"type": "wandb", "wandb": {"project": "gpt-pricer"}}

# Example job creation (actual job_id needs to be retrieved from response)
job_creation_response = openai.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=validation_file.id,
    model="gpt-4o-mini-2024-07-18", # Specify the base model
    seed=42,
    hyperparameters={"n_epochs": 1},
    integrations = [wandb_integration],
    suffix="pricer"
)
job_id = job_creation_response.id

# To retrieve the fine-tuned model name after the job completes:
# fine_tuned_model_name = openai.fine_tuning.jobs.retrieve(job_id).fine_tuned_model
```

**Testing the Fine-tuned Model:**
```python
# Assuming fine_tuned_model_name is available after fine-tuning
def gpt_fine_tuned(item):
    response = openai.chat.completions.create(
        model=fine_tuned_model_name, # Use the fine-tuned model
        messages=messages_for(item),
        seed=42,
        max_tokens=7 # Allow slightly more tokens for fine-tuned output
    )
    reply = response.choices[0].message.content
    return get_price(reply)

# Tester.test(gpt_fine_tuned, test)
```