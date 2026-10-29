import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


class ResponseGenerator:
    def __init__(self, model_id="models/TinyLlama-1.1B-Chat-v1.0"):
        if not os.path.isdir(model_id):
            raise FileNotFoundError(
                f"Local LLM folder not found: {model_id}. "
                "Please download TinyLlama/TinyLlama-1.1B-Chat-v1.0 and place it inside the models folder."
            )

        if not os.path.exists(os.path.join(model_id, "config.json")):
            raise FileNotFoundError(
                f"config.json not found inside {model_id}. "
                "The TinyLlama folder is incomplete. Download the full model repository."
            )

        print(f"Loading local LLM from: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            dtype=torch.float32
        )

        self.model.eval()
    
    def clean_citations_and_questions(self, text):
        # Remove square bracket numeric citations like [88], [90, 91]
        text = re.sub(r'\[\s?\d+(?:\s?,\s?\d+)*\s?\]', '', text)
        
        # Remove full structured reference-style citations like:
        # "Smith AB. J Clin Med. 2021;20(2):101–105."
        text = re.sub(
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\.?\s*)?(?:[A-Za-z\s]+\.)?\s?\d{4};\d+\(.*?\):\d+(–\d+)?\.',
            '', text
        )
        
        text = re.sub(r'\b\w+\s+et al\.,?\s*\d{0,4}', '', text)
        # Clean extra spacing artifacts
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def generate_response(self, query, context):
        escaped = context.strip()
        prompt = f"""You are a medical assistant trained in PCOS. Answer the QUESTION using only the information provided in the CONTEXT below.
Do not use any outside knowledge or make up information. Be accurate, concise, and factual.

QUESTION: {query}
CONTEXT: {escaped}
ANSWER:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]
            max_total_tokens = getattr(self.model.config, "max_position_embeddings", self.model.config.max_position_embeddings)
            # Dynamically allocate max_new_tokens based on input length
            max_new_tokens = max(32, min(512, max_total_tokens - input_len))
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,do_sample=False,repetition_penalty=1.1
            )
            full_output  = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the decoded output (optional)
            answer = full_output.split("ANSWER:")[-1].strip()
            answer = self.clean_citations_and_questions(answer)
            return answer
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, something went wrong."

    def generate_without_context(self, query):
        """
        Generator-only baseline (no retrieval).
        """
        prompt = f"""You are a medical assistant trained in PCOS.

    QUESTION: {query}

    ANSWER:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {
                k: v.to(self.model.device)
                for k, v in inputs.items()
            }

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1
            )

            full_output = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            answer = full_output.split("ANSWER:")[-1].strip()
            answer = self.clean_citations_and_questions(answer)

            return answer

        except Exception as e:
            print(f"Generator-only Error: {e}")
            return "Sorry, something went wrong."