import os
import sys
import torch
import numpy as np
import json
import evaluate
import wandb
import re  
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from tqdm import tqdm
from loguru import logger
import gc

# Configure memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logger.remove()
logger.add("eval_log_test2.log", level="INFO", backtrace=True, diagnose=True)
logger.add(lambda msg: print(msg), level="INFO")

# Set up exception handler
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.error(f"Uncaught exception: {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
sys.excepthook = handle_exception

def main():
    try:
        # Configure device
        if torch.cuda.is_available():
            device = "cuda"
            # Force garbage collection to free up memory
            torch.cuda.empty_cache()
            gc.collect()
        else:
            device = "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Configuration
        checkpoint_path = "llama-3-8b-mental_health-chat_optuna_run_3/checkpoint-22500"
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
        torch_dtype = torch.float16
        batch_size = 1  
        
        # Load metrics
        rouge_metric = evaluate.load("rouge")
        bertscore_metric = evaluate.load("bertscore")
        
        # Initialize wandb for logging
        wandb.init(
            project="llama3-mental-health-eval",
            name="main_optuna_run_evaluation_1",
            config={
                "checkpoint": checkpoint_path,
                "base_model": base_model,
                "batch_size": batch_size,
            }
        )
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Set padding token for the tokenizer
        if tokenizer.pad_token is None:
            logger.info("Setting padding token to EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "right"
        
        # Load base model with memory optimizations
        logger.info("Loading base model with memory optimizations...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            torch_dtype=torch_dtype,
            use_cache=False,
        )
        
        # Set pad token ID in the model config as well
        if model.config.pad_token_id is None:
            logger.info("Setting model pad_token_id to eos_token_id")
            model.config.pad_token_id = model.config.eos_token_id
        
        # Load the adapter weights
        logger.info(f"Loading adapter from {checkpoint_path}...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        
        # Dataset processing function
        def format_chat_template(row):
            try:
                turns = json.loads(row["text"])
                messages = [
                    {"role": "user" if speaker == "usr" else "assistant", "content": message}
                    for speaker, message in turns
                ]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                return {"text": formatted}
            except Exception as e:
                logger.error(f"Error in format_chat_template: {e}")
                return {"text": ""}
        
        # Load and prepare dataset - ONLY USE THE TEST SPLIT FROM ORIGINAL TRAINING
        logger.info("Loading dataset using the same split as in training...")
        dataset = load_dataset("thu-coai/augesc")
        
        # Use the same seed and selection as in training
        selected_dataset = dataset["train"].shuffle(seed=42).select(range(15000))
        
        # Apply the same train/test split as in training (80/20)
        train_test_split = selected_dataset.train_test_split(test_size=0.2, seed=42)
        
        # Extract only the test split for evaluation
        test_dataset = train_test_split["test"]
        
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Process the test dataset
        test_dataset = test_dataset.map(format_chat_template, num_proc=4)
        
        # Log a few samples to understand format
        for i in range(min(3, len(test_dataset))):
            logger.info(f"Sample {i} raw text: {test_dataset[i]['text'][:300]}...")
        
        # Function for metrics calculation
        def calculate_metrics(predictions, references, user_contexts=None):
            # Filter out empty references and placeholder references which cause warnings in BERTScore
            valid_pairs = [(pred, ref) for pred, ref in zip(predictions, references) 
                          if ref.strip() and not ref.endswith("[No valid reference available]") 
                          and not ref.endswith("[Error extracting reference]")]
            
            # Log the number of valid pairs
            logger.info(f"Found {len(valid_pairs)} valid reference-prediction pairs out of {len(predictions)} total")
            
            # Skip metrics calculation if no valid pairs
            if not valid_pairs:
                logger.warning("No valid reference-prediction pairs found for metrics")
                return {
                    "rouge_rouge1": 0.0, "rouge_rouge2": 0.0, "rouge_rougeL": 0.0, "rouge_rougeLsum": 0.0,
                    "bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0
                }
                
            # Unzip the valid pairs
            valid_predictions, valid_references = zip(*valid_pairs)
            
            # ROUGE
            rouge_result = rouge_metric.compute(
                predictions=valid_predictions, references=valid_references, use_stemmer=True
            )
            rouge_result = {f"rouge_{k}": v * 100 for k, v in rouge_result.items()}
            
            # BERTScore 
            bs_batch_size = 8
            all_precision = []
            all_recall = []
            all_f1 = []
            
            for i in range(0, len(valid_predictions), bs_batch_size):
                batch_preds = valid_predictions[i:i+bs_batch_size]
                batch_refs = valid_references[i:i+bs_batch_size]
                
                bert_result = bertscore_metric.compute(
                    predictions=batch_preds, references=batch_refs, lang="en"
                )
                
                all_precision.extend(bert_result["precision"])
                all_recall.extend(bert_result["recall"])
                all_f1.extend(bert_result["f1"])
                
                # Force garbage collection after each batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            bert_avg = {
                "bertscore_precision": np.mean(all_precision) * 100,
                "bertscore_recall": np.mean(all_recall) * 100,
                "bertscore_f1": np.mean(all_f1) * 100,
            }
            
            # Calculate additional open-source metrics for mental health conversations
            
            # 1. Toxicity detection using detoxify
            toxicity_scores = calculate_toxicity(valid_predictions)
            
            # 2. Sentiment analysis to measure response tone
            sentiment_scores = calculate_sentiment(valid_predictions, valid_references)
            
            # 3. Lexical diversity metrics
            diversity_metrics = calculate_lexical_diversity(valid_predictions)
            
            # 4. Mental health empathy metrics using custom keyword-based approach
            empathy_metrics = calculate_empathy_metrics(valid_predictions, user_contexts)
            
            # Combine all metrics
            all_metrics = {
                **rouge_result, 
                **bert_avg, 
                **toxicity_scores,
                **sentiment_scores,
                **diversity_metrics,
                **empathy_metrics
            }
            
            # Log how many valid pairs were used for metrics
            logger.info(f"Calculated metrics using {len(valid_predictions)} valid reference-prediction pairs")
            
            return all_metrics

        def calculate_toxicity(predictions):
            """Calculate toxicity scores for predictions using Detoxify."""
            try:
                from detoxify import Detoxify
                import numpy as np
                import warnings
                
                # Suppress the specific warning about uninitialized weights
                warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized")
                
                # Process in batches to save memory
                batch_size = 8
                toxicity_scores = []
                severe_toxicity_scores = []
                
                # Initialize Detoxify model
                detox_model = Detoxify('original')
                
                for i in range(0, len(predictions), batch_size):
                    batch = predictions[i:min(i+batch_size, len(predictions))]
                    
                    # Get toxicity scores
                    results = detox_model.predict(batch)
                    
                    toxicity_scores.extend(results['toxicity'])
                    severe_toxicity_scores.extend(results['severe_toxicity'])
                    
                    # Free memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate average scores
                avg_toxicity = np.mean(toxicity_scores) * 100
                avg_severe_toxicity = np.mean(severe_toxicity_scores) * 100
                
                # Safety score = 100 - toxicity (to make higher = better)
                safety_score = 100 - avg_toxicity
                
                logger.info(f"Toxicity analysis completed on {len(toxicity_scores)} texts")
                
                return {
                    "safety_score": safety_score,
                    "toxicity_score": avg_toxicity,
                    "severe_toxicity_score": avg_severe_toxicity
                }
                
            except Exception as e:
                logger.error(f"Error in toxicity calculation: {e}")
                return {
                    "safety_score": 0.0,
                    "toxicity_score": 0.0,
                    "severe_toxicity_score": 0.0
                }

        def calculate_sentiment(predictions, references):
            """Calculate sentiment metrics for predictions vs references."""
            try:
                from transformers import pipeline
                import numpy as np
                
                # Initialize sentiment analysis pipeline
                sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Process in batches
                batch_size = 8
                pred_positive_scores = []
                ref_positive_scores = []
                
                for i in range(0, len(predictions), batch_size):
                    pred_batch = predictions[i:min(i+batch_size, len(predictions))]
                    ref_batch = references[i:min(i+batch_size, len(references))]
                    
                    # Process each text individually with truncation
                    for j in range(len(pred_batch)):
                        # Truncate long texts to 512 characters to avoid token length issues
                        pred_text = pred_batch[j][:512]
                        ref_text = ref_batch[j][:512]
                        
                        try:
                            # Get sentiment scores
                            pred_result = sentiment_analyzer(pred_text)[0]
                            ref_result = sentiment_analyzer(ref_text)[0]
                            
                            # Extract positive scores (convert negative to 0, positive to score)
                            if pred_result['label'] == 'POSITIVE':
                                pred_positive_scores.append(pred_result['score'])
                            else:
                                pred_positive_scores.append(1 - pred_result['score'])
                                
                            if ref_result['label'] == 'POSITIVE':
                                ref_positive_scores.append(ref_result['score'])
                            else:
                                ref_positive_scores.append(1 - ref_result['score'])
                        except Exception as e:
                            logger.warning(f"Error processing sentiment for text: {e}")
                            continue
                    
                    # Free memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Calculate average scores
                avg_pred_sentiment = np.mean(pred_positive_scores) * 100 if pred_positive_scores else 0
                avg_ref_sentiment = np.mean(ref_positive_scores) * 100 if ref_positive_scores else 0
                
                # Sentiment alignment (higher = closer to reference sentiment)
                sentiment_alignment = 100 - abs(avg_pred_sentiment - avg_ref_sentiment)
                
                logger.info(f"Sentiment analysis completed on {len(pred_positive_scores)} texts")
                
                return {
                    "response_positivity": avg_pred_sentiment,
                    "reference_positivity": avg_ref_sentiment,
                    "sentiment_alignment": sentiment_alignment
                }
                
            except Exception as e:
                logger.error(f"Error in sentiment calculation: {e}")
                return {
                    "response_positivity": 0.0,
                    "reference_positivity": 0.0,
                    "sentiment_alignment": 0.0
                }

        def calculate_lexical_diversity(texts):
            """Calculate lexical diversity metrics for a list of texts."""
            try:
                import nltk
                import numpy as np
                import re
                
                # Type-Token Ratio (TTR) calculation
                ttrs = []
                sentence_lengths = []
                
                for text in texts:
                    # Simple tokenization using regex instead of nltk
                    tokens = re.findall(r'\b\w+\b', text.lower())
                    
                    # Calculate TTR if we have tokens
                    if tokens:
                        unique_tokens = set(tokens)
                        ttr = len(unique_tokens) / len(tokens)
                        ttrs.append(ttr)
                    
                    # Simple sentence splitting by punctuation
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if sentence.strip():  # Only non-empty sentences
                            words = re.findall(r'\b\w+\b', sentence.lower())
                            if words:
                                sentence_lengths.append(len(words))
                
                avg_ttr = np.mean(ttrs) * 100 if ttrs else 0
                avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
                
                logger.info(f"Lexical diversity calculated on {len(ttrs)} texts")
                
                return {
                    "lexical_diversity_ttr": avg_ttr,
                    "avg_sentence_length": avg_sentence_length
                }
                
            except Exception as e:
                logger.error(f"Error in lexical diversity calculation: {e}")
                return {
                    "lexical_diversity_ttr": 0.0,
                    "avg_sentence_length": 0.0
                }

        def calculate_empathy_metrics(predictions, user_contexts=None):
            """
            Calculate empathy metrics for mental health responses using keyword-based approach.
            This is a simple open-source approximation of empathy measurement.
            """
            try:
                import re
                import numpy as np
                
                # Lists of empathy-indicating phrases and patterns
                empathy_phrases = [
                    r"\bi understand\b", r"\bi hear you\b", r"\bthat sounds\b",
                    r"\bi can imagine\b", r"\bmust be\b", r"\bit seems\b",
                    r"\byou feel\b", r"\byour feelings\b", r"feeling", r"emotions?",
                    r"\bi'm sorry\b", r"\bi am sorry\b", r"\bsorry to hear\b",
                    r"\bdifferent for everyone\b", r"\byour experience\b", r"\bthank you for sharing\b",
                    r"\bmake sense\b", r"\bunderstandable\b", r"\bvalid\b"
                ]
                
                # Lists of engagement-indicating phrases
                engagement_phrases = [
                    r"\bhave you tried\b", r"\byou could try\b", r"\bwould you consider\b",
                    r"\bwhat about\b", r"\bhave you thought about\b", r"\bcould be helpful\b",
                    r"\bmight help\b", r"\bsuggestion\b", r"\brecommend\b", r"\boption\b",
                    r"\bstrategy\b", r"\btechnique\b", r"\bapproach\b", r"\bexercise\b",
                    r"\bprofessional\b", r"\btherapist\b", r"\bcounselor\b", r"\bsupport group\b",
                    r"\bresources?\b"
                ]
                
                # Lists of personal experiences indicators
                personal_phrases = [
                    r"\bmany people\b", r"\bsome people\b", r"\bothers have\b", 
                    r"\bit's common\b", r"\bit is common\b", r"\bnot alone\b",
                    r"\bother people\b", r"\bmany individuals\b", r"\bshared experience\b"
                ]
                
                # Question indicators for engagement
                question_indicators = [
                    r"\?", r"\bwhat\b.*\?", r"\bhow\b.*\?", r"\bwhen\b.*\?", 
                    r"\bwhere\b.*\?", r"\bwhy\b.*\?", r"\bdo you\b.*\?", r"\bare you\b.*\?"
                ]
                
                empathy_scores = []
                engagement_scores = []
                normalization_scores = []
                question_counts = []
                
                for text in predictions:
                    # Convert to lowercase for matching
                    lower_text = text.lower()
                    
                    # Count empathy phrases
                    empathy_count = sum(1 for phrase in empathy_phrases if re.search(phrase, lower_text))
                    
                    # Count engagement phrases
                    engagement_count = sum(1 for phrase in engagement_phrases if re.search(phrase, lower_text))
                    
                    # Count normalizing phrases
                    normal_count = sum(1 for phrase in personal_phrases if re.search(phrase, lower_text))
                    
                    # Count questions (engagement indicator)
                    question_count = sum(1 for q in question_indicators if re.search(q, lower_text))
                    
                    # Calculate scores (normalized by text length to avoid favoring longer texts)
                    word_count = len(lower_text.split())
                    norm_factor = max(1, word_count / 20)  # Normalize by 20 words
                    
                    empathy_scores.append(min(10, empathy_count / norm_factor))
                    engagement_scores.append(min(10, engagement_count / norm_factor))
                    normalization_scores.append(min(10, normal_count / norm_factor))
                    question_counts.append(question_count)
                
                # Calculate average scores and scale to 0-100
                avg_empathy = np.mean(empathy_scores) * 10 if empathy_scores else 0
                avg_engagement = np.mean(engagement_scores) * 10 if engagement_scores else 0
                avg_normalization = np.mean(normalization_scores) * 10 if normalization_scores else 0
                avg_questions = np.mean(question_counts) * 10 if question_counts else 0
                
                # Compound empathy score
                empathy_combined = (avg_empathy + avg_normalization) / 2
                
                return {
                    "empathy_score": empathy_combined,
                    "actionable_guidance": avg_engagement,
                    "normalization_score": avg_normalization,
                    "question_engagement": avg_questions
                }
                
            except Exception as e:
                logger.error(f"Error in empathy metrics calculation: {e}")
                return {
                    "empathy_score": 0.0,
                    "actionable_guidance": 0.0,
                    "normalization_score": 0.0,
                    "question_engagement": 0.0
                }
        
        # Evaluation function that works in smaller batches
        def evaluate_model():
            model.eval()
            all_predictions = []
            all_references = []
            
            # Process the test dataset in batches
            for i in tqdm(range(0, len(test_dataset), batch_size)):
                batch = test_dataset.select(range(i, min(i + batch_size, len(test_dataset))))
                
                # Get input text
                inputs = batch["text"]
                
                # Get only the user's part for input, assistant's part for reference
                references = []
                processed_inputs = []
                
                # Add logging to understand data format
                if i == 0:
                    logger.info(f"Sample input text format: {inputs[0][:500]}...")
                
                # Count valid references for this batch
                valid_refs_count = 0
                
                for text in inputs:
                    try:
                        # Extract assistant's response using the special tokens format in the dataset
                        # Pattern: <|start_header_id|>assistant<|end_header_id|> followed by the response 
                        # then potentially <|eot_id|>
                        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
                        eot_marker = "<|eot_id|>"
                        
                        if assistant_marker in text:
                            last_assistant_idx = text.rfind(assistant_marker)
                            
                            # For input, include everything up to and including the assistant marker
                            processed_input = text[:last_assistant_idx + len(assistant_marker)]
                            
                            # Extract the reference - what comes after assistant marker
                            reference_text = text[last_assistant_idx + len(assistant_marker):]
                            
                            # If there's an EOT marker after the assistant response, trim it
                            if eot_marker in reference_text:
                                reference_text = reference_text.split(eot_marker)[0]
                                
                            # Only use if the assistant actually said something
                            if reference_text.strip():
                                processed_inputs.append(processed_input)
                                references.append(reference_text.strip())
                                valid_refs_count += 1
                                
                                if i < 2:  # Log a few samples
                                    logger.info(f"Valid reference found: '{reference_text.strip()[:50]}...'")
                                continue
                        
                        # Log when can't parse properly
                        if i < 5:  # Only log a few samples
                            logger.debug(f"Couldn't extract valid reference from: '{text[:100]}...'")
                        
                        # Default case if can't extract properly
                        processed_inputs.append(text)
                        # Use a special placeholder that won't be counted in metrics
                        references.append("[No valid reference available]")
                        
                    except Exception as e:
                        logger.error(f"Error splitting conversation: {e}")
                        processed_inputs.append(text)
                        references.append("[Error extracting reference]")
                
                logger.info(f"Batch {i//batch_size}: Found {valid_refs_count} valid references out of {len(inputs)} inputs")
                
                # Tokenize inputs
                tokenized_inputs = tokenizer(
                    processed_inputs, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1024
                ).to(device)
                
                # Generate with memory optimizations
                with torch.inference_mode():
                    generated_ids = model.generate(
                        **tokenized_inputs,
                        max_new_tokens=512,
                        do_sample=True,  
                        temperature=0.6,  
                        top_p=0.9,       
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # Decode generated text
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Add to collections
                all_predictions.extend(decoded_preds)
                all_references.extend(references)
                
                # Force garbage collection after each batch
                del tokenized_inputs, generated_ids, decoded_preds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate metrics on the collected predictions and references
            metrics = calculate_metrics(all_predictions, all_references)
            
            return metrics, all_predictions, all_references
        
        # Run evaluation
        logger.info("Starting evaluation...")
        metrics, predictions, references = evaluate_model()
        
        # Log results
        logger.info(f"Evaluation Results: {metrics}")
        wandb.log(metrics)
        
        # Save sample predictions
        sample_results = []
        for i in range(min(10, len(predictions))):
            sample_results.append({
                "prediction": predictions[i],
                "reference": references[i],
            })
        
        with open("sample_predictions.json", "w") as f:
            json.dump(sample_results, f, indent=2)
        
        wandb.save("sample_predictions.json")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()