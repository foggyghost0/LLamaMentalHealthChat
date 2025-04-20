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
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
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

# Custom callback for Optuna pruning
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial, monitor="eval_loss"):
        self.trial = trial
        self.monitor = monitor
        
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self.monitor in metrics:
            self.trial.report(metrics[self.monitor], state.global_step)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

# Configure logging properly for loguru
logger.remove() 
logger.add("log.log", level="INFO", backtrace=True, diagnose=True)  
logger.add(lambda msg: print(msg), level="INFO")

# Set up exception handler to log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error(f"Uncaught exception: {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default exception handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
sys.excepthook = handle_exception

try:
    # Dynamically set the GPU device to index 1 or fallback to CPU
    available_gpus = torch.cuda.device_count()
    if available_gpus > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to GPU index 1
        device = "cuda:0"  # When using CUDA_VISIBLE_DEVICES, the first visible device becomes cuda:0
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")

    # Load custom dataset
    try:
        dataset = load_dataset("thu-coai/augesc")
        logger.info("Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    # Configuration
    base_model = "meta-llama/Llama-3.1-8B-Instruct"
    new_model = "llama-3-8b-mental_health-chat_optuna_run_3"
    torch_dtype = torch.float16
    attn_implementation = "eager"

    # Initialize WandB
    try:
        wandb.login(key="")
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
            device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu",
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = "right"  
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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
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
    def format_chat_template(row):
        try:
            # Parse the stringified list of lists into actual Python list
            turns = json.loads(row["text"])

            # Convert to expected format for chat template
            messages = [
                {"role": "user" if speaker == "usr" else "assistant", "content": message}
                for speaker, message in turns
            ]

            # Apply chat template
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Return a dict
            return {"text": formatted}
        except Exception as e:
            logger.error(f"Error in format_chat_template: {e}")
            # Return an empty but valid result to avoid breaking the mapping
            return {"text": ""}



    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=15,
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

    # Dataset processing - apply chat template and split
    try:
        dataset = load_dataset("thu-coai/augesc")
        dataset = dataset["train"].shuffle(seed=42).select(range(15000))
        dataset = dataset.map(format_chat_template, num_proc=4)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        logger.info("Dataset processed and split successfully")
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

    # Save the test set for final evaluation (unseen during HPO)
    test_dataset_final = dataset["test"]
    train_dataset_full = dataset["train"]

    # Optuna objective function

    def objective(trial):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.cuda.empty_cache()

        # Sample hyperparameters (only learning_rate and batch_size)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [2, 4, 6, 8])

        # Reload model and tokenizer for each trial
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu",
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.padding_side = "right"
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        # Use different random subsets for each trial (from train split only)
        train_indices = np.random.choice(len(train_dataset_full), size=1000, replace=False)
        eval_indices = np.random.choice(list(set(range(len(train_dataset_full))) - set(train_indices)), size=200, replace=False)
        train_subset = train_dataset_full.select(train_indices)
        eval_subset = train_dataset_full.select(eval_indices)

        # Training arguments for trial (no saving, 6 epochs)
        trial_args = TrainingArguments(
            output_dir="/tmp/optuna-trash",  # dummy dir, will not be used
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
            optim=training_arguments.optim,
            num_train_epochs=6,
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
            callbacks=[OptunaPruningCallback(trial)],
        )

        # Training
        trainer.train()
        eval_results = trainer.evaluate()
        torch.cuda.empty_cache()
        return eval_results["eval_loss"]

    # Optuna study with TPE sampler
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    # Best hyperparameters
    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    # Final training with best hyperparameters and full train set
    final_args = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        per_device_eval_batch_size=best_params['per_device_train_batch_size'],
        gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        optim=training_arguments.optim,
        num_train_epochs=15,  
        eval_steps=training_arguments.eval_steps,
        logging_steps=training_arguments.logging_steps,
        learning_rate=best_params['learning_rate'],
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
        device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu",
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    class VerboseEvalLogger(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            logger.info(f"[FINAL TRAINING][EVAL] Step {state.global_step}: {metrics}")

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
        trainer.model.save_pretrained(new_model)
        tokenizer.save_pretrained(new_model)
        logger.info(f"Model saved to {new_model}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


except Exception as e:
    logger.error(f"Unexpected error in main execution: {e}")
    logger.error(traceback.format_exc())