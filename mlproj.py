import os
from datasets import load_dataset
import json
import traceback
import sys

import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer
import numpy as np
from loguru import logger
import optuna
from torch.utils.data import Subset
from huggingface_hub import login

login(token="SECRET")

# Custom callback for Optuna pruning
class OptunaPruningCallback(
    TrainerCallback
): 
    def __init__(self, trial, monitor="eval_loss"):
        self.trial = trial  
        self.monitor = monitor  

    def on_evaluate(
        self, args, state, control, **kwargs
    ):  
        metrics = kwargs.get("metrics", {})
        if self.monitor in metrics:
            self.trial.report(
                metrics[self.monitor], state.global_step
            )  
            if (
                self.trial.should_prune()
            ): 
                raise optuna.TrialPruned()  

# Configure logging properly for loguru
logger.remove()  # removes the default logger that Loguru sets up automatically
logger.add(
    "log_final_big.log", level="INFO", backtrace=True, diagnose=True
) 
logger.add(
    lambda msg: print(msg), level="INFO"
) 

# Set up exception handler to log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error(
        f"Uncaught exception: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = handle_exception

try:
    device = "cuda"

    logger.info(f"Using device: {device}")

    # Load custom dataset
    try:
        dataset = load_dataset("thu-coai/augesc")
        logger.info("Dataset loaded successfully")
        # Ensure we are working with a Dataset, not a list or IterableDataset
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
        # If dataset is an IterableDataset, raise an error or convert if possible
        if not hasattr(dataset, "shuffle") or not hasattr(dataset, "select"):
            raise RuntimeError(
                "Loaded dataset is not a HuggingFace Dataset with shuffle/select support."
            )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Configuration
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    new_model = "llama-3-8b-mental_health-chat_optuna_run_final"
    
    torch_dtype = (
        torch.float16
    )  # sets the data type for the model weights to float16 (half-precision) -> reduces memory usage and speeds up computation
    attn_implementation = (
        "eager"  # using PyTorch’s eager execution mode (standard attention mechanism)
    )

    # Initialize WandB
    try:
        wandb.login(key="752eed432a980ec5bc2af208d67e5d0ef890ef53")
        logger.info("WandB login successful")
    except Exception as e:
        logger.error(f"WandB login failed: {e}")
        raise

    # QLoRA Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load Model and Tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=(
                {"": torch.cuda.current_device()}
                if torch.cuda.is_available()
                else "cpu"
            ),
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  
        model.config.use_cache = False
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise

    # Prepare model
    try:
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for kbit training")
    except Exception as e:
        logger.error(f"Error preparing model for kbit training: {e}")
        raise

    # LoRA Configuration
    peft_config = LoraConfig(
        r=16,  # rank (number of trainable parameters)
        lora_alpha=32,  # scaling factor for the LoRA parameters
        lora_dropout=0.05,  # dropout applied to the LoRA layers for regularization
        bias="none",  # do not train biases; only LoRA layers are trainable
        task_type="CAUSAL_LM",
        target_modules=[  # Specifies which layers to inject LoRA adapters into
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    try:
        model = get_peft_model(model, peft_config)
        logger.info("PEFT model created successfully")
    except Exception as e:
        logger.error(f"Error creating PEFT model: {e}")
        raise

    # Dataset processing
    # Optimized strict_causal_examples function for batch processing
    def strict_causal_examples_batched(batch):
        all_input_strings = []
        all_assistant_turns = []
        example_indices = []

        for i, row in enumerate(batch["text"]):
            try:
                turns = json.loads(row)
                messages = [
                    {
                        "role": "user" if speaker == "usr" else "assistant",
                        "content": message,
                    }
                    for speaker, message in turns
                ]

                for idx, (speaker, message) in enumerate(turns):
                    if speaker != "sys" and speaker != "usr":
                        continue
                    if speaker == "usr":
                        continue # Only create examples for assistant turns

                    msg_idx = sum(1 for s, _ in turns[: idx + 1] if s != "sys")
                    history = messages[:msg_idx]

                    if history and history[-1]["role"] == "assistant":
                        formatted = tokenizer.apply_chat_template(
                            history, tokenize=False, add_generation_prompt=False
                        )
                        all_input_strings.append(formatted)
                        all_assistant_turns.append(history[-1])
                        example_indices.append(i) # Store the index of the original row

            except Exception as e:
                logger.error(f"Error processing row {i} in strict_causal_examples_batched: {e}")
                continue

        if not all_input_strings:
             return {"input_ids": [], "attention_mask": [], "labels": []}


        # Tokenize all collected strings in one go
        tokenized_results = tokenizer(
            all_input_strings,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
        )

        # Create labels based on the tokenized results
        all_labels = []
        for i in range(len(all_input_strings)):
            assistant_turn = all_assistant_turns[i]
            assistant_tokens = tokenizer(
                tokenizer.apply_chat_template(
                    [assistant_turn],
                    tokenize=False,
                    add_generation_prompt=False,
                ),
                add_special_tokens=False,
            )["input_ids"]

            input_ids = tokenized_results["input_ids"][i]
            label = [-100] * (len(input_ids) - len(assistant_tokens)) + assistant_tokens
            if len(label) < tokenizer.model_max_length:
                label += [-100] * (tokenizer.model_max_length - len(label))
            else:
                label = label[: tokenizer.model_max_length]

            all_labels.append(label)

        return {
            "input_ids": tokenized_results["input_ids"],
            "attention_mask": tokenized_results["attention_mask"],
            "labels": all_labels,
        }


    training_arguments = TrainingArguments(
        output_dir=new_model, 
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        gradient_accumulation_steps=2,  
        optim="paged_adamw_32bit",  
        num_train_epochs=12, 
        eval_steps=1000, 
        logging_steps=1000, 
        learning_rate=5e-5,
        lr_scheduler_type="cosine", 
        fp16=True,  
        bf16=False,
        group_by_length=True,  
        report_to="wandb", 
        save_strategy="steps",
        save_steps=1000,  
        warmup_ratio=0.03, 
    )


#  Dataset processing - strict causal SFT
    try:
        dataset = dataset.shuffle(seed=42).select(range(10000))
        logger.info(f"Original dataset size after selection: {len(dataset)}")

        chunk_size = 500  # Adjust chunk size based on available memory
        processed_chunks = []

        for i in range(0, len(dataset), chunk_size):
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(dataset) + chunk_size - 1) // chunk_size}")
            chunk = dataset.select(range(i, min(i + chunk_size, len(dataset))))
            tokenized_chunk = chunk.map(
                strict_causal_examples_batched, 
                batched=True,
                remove_columns=chunk.column_names,
                num_proc=os.cpu_count(),
            )
            processed_chunks.append(tokenized_chunk)

        # Combine all processed chunks into a single Dataset
        from datasets import concatenate_datasets
        tokenized_dataset = concatenate_datasets(processed_chunks)

        # Split into train/val
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        tokenized_train = tokenized_dataset["train"]
        tokenized_val = tokenized_dataset["test"]
        logger.info("Dataset processed for strict causal SFT in chunks and split successfully")
        logger.info(f"Tokenized train dataset size: {len(tokenized_train)}")
        logger.info(f"Tokenized validation dataset size: {len(tokenized_val)}")

    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

    # Save the val set for final evaluation (unseen during HPO)
    test_dataset_final = tokenized_val
    train_dataset_full = tokenized_train

    # Optuna objective function

    def objective(
        trial,
    ):  # function is called for each trial during the Optuna optimization
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.cuda.empty_cache()  # clear GPU memory before each trial

        # Sample hyperparameters (only learning_rate and batch_size)
        learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ) 
        per_device_train_batch_size = trial.suggest_categorical(
            "per_device_train_batch_size",
            [2, 4, 6, 8],  # Optuna chooses a batch size from the given list
        )

        # Reload model and tokenizer for each trial
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map=(
                {"": torch.cuda.current_device()}
                if torch.cuda.is_available()
                else "cpu"
            ),
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        # Use different random subsets for each trial (from train split only)
        # Selects a random 1000-sample training subset and 200-sample eval subset from the train dataset
        # This is done to avoid overfitting to a specific subset of the data during hyperparameter tuning
        # Keeps each trial fast and reproducible
        train_indices = np.random.choice(
            len(train_dataset_full), size=1000, replace=False
        )
        eval_indices = np.random.choice(
            list(set(range(len(train_dataset_full))) - set(train_indices)),
            size=200,
            replace=False,
        )
        train_subset = train_dataset_full.select(train_indices)
        eval_subset = train_dataset_full.select(eval_indices)

        # Training arguments for trial (no saving, 6 epochs)
        trial_args = TrainingArguments(
            output_dir="/tmp/optuna-trash",  # dummy dir, will not be used
            per_device_train_batch_size=per_device_train_batch_size,  
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
            optim=training_arguments.optim,
            num_train_epochs=5,
            eval_steps=100,
            logging_steps=100,
            learning_rate=learning_rate,  
            lr_scheduler_type=training_arguments.lr_scheduler_type, 
            fp16=training_arguments.fp16,
            bf16=training_arguments.bf16,
            group_by_length=training_arguments.group_by_length,
            report_to=[],  # Disable wandb for HPO
            save_strategy="no",  # Do not save trial models
            warmup_ratio=training_arguments.warmup_ratio,
        )

        # Trainer
        trainer = SFTTrainer(  
            model=model,
            train_dataset=train_subset, 
            eval_dataset=eval_subset,  
            peft_config=peft_config,
            args=trial_args, 
            callbacks=[
                OptunaPruningCallback(trial)
            ],  
        )

        # Training
        trainer.train()  
        eval_results = trainer.evaluate()  
        torch.cuda.empty_cache() 
        return eval_results[
            "eval_loss"
        ]  

    # Optuna study with TPE sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=12)

    # Best hyperparameters
    best_params = study.best_params
    logger.info(
        f"Best hyperparameters: {best_params}"
    )  

    # Final training with best hyperparameters and full train set
    final_args = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=best_params[
            "per_device_train_batch_size"
        ], 
        per_device_eval_batch_size=best_params[
            "per_device_train_batch_size"
        ], 
        gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        optim=training_arguments.optim,
        num_train_epochs=12,
        eval_steps=training_arguments.eval_steps,
        logging_steps=training_arguments.logging_steps,
        learning_rate=best_params["learning_rate"],  
        lr_scheduler_type=training_arguments.lr_scheduler_type,
        fp16=training_arguments.fp16,
        bf16=training_arguments.bf16,
        group_by_length=training_arguments.group_by_length,
        report_to="wandb",
        save_strategy=training_arguments.save_strategy,
        save_steps=training_arguments.save_steps,
        warmup_ratio=training_arguments.warmup_ratio,
    )

    # Reload model/tokenizer for final training
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=(
            {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
        ),
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    class VerboseEvalLogger(
        TrainerCallback
    ):  
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics", {})
            logger.info(
                f"[FINAL TRAINING][EVAL] Step {state.global_step}: {metrics}"
            )  

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_full,
        eval_dataset=test_dataset_final,
        peft_config=peft_config,
        args=final_args,
        callbacks=[VerboseEvalLogger()],
    )

    try:
        trainer.train()
        logger.info("Final training completed successfully")
    except Exception as e:
        logger.error(f"Error during final training: {e}")
        raise

    try:
        trainer.save_model(new_model)
        tokenizer.save_pretrained(new_model)
        logger.info(f"Model saved to {new_model}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


except Exception as e:
    logger.error(f"Unexpected error in main execution: {e}")
    logger.error(traceback.format_exc())
