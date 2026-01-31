
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import logging

logger = logging.getLogger(__name__)

class LLMCandidateGenerator:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
        logger.info(f"Loading LLM: {model_id}")
        if torch.cuda.is_available():
            logger.info(f"CUDA Available. Device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA NOT AVAILABLE. Running on CPU!")

        # Load in 4-bit to save memory (requires roughly 6GB VRAM)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Explicitly trust remote code if needed (Llama usually doesn't need it but good practice)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        logger.info(f"Model Device Map: {self.model.hf_device_map}")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.9,
            # device_map="auto" is handled by model
        )
        logger.info("LLM loaded successfully.")

    def generate_candidates(self, query):
        # The Exact Prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Information Retrieval Assistant. 
Task: Expand the user query into a rich set of keywords and a pseudo-passage.
Return ONLY valid JSON.

Format:
{{
  "entities": ["entity1", "entity2"],
  "synonyms": ["synonym1", "synonym2"],
  "passage": "Full dense passage here."
}}
<|eot_id|><|start_header_id|>user<|end_header_id|>

Query: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # Generate
        outputs = self.pipe(
            prompt, 
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Parse Output (strip the prompt)
        generated_text = outputs[0]['generated_text'][len(prompt):]
        
        try:
            # Simple heuristic to find JSON start/end if there's noise
            start = generated_text.find('{')
            end = generated_text.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = generated_text[start:end]
                data = json.loads(json_str)
            else:
                data = json.loads(generated_text)
            
            # COMBINE EVERYTHING for the Harvester
            entities = data.get('entities', [])
            synonyms = data.get('synonyms', [])
            passage = data.get('passage', "")
            
            combined_text = f"{' '.join(entities)} {' '.join(synonyms)} {passage}"
            return combined_text, data 
            
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM JSON decode error: {e}. Output: {generated_text[:100]}...")
            # Fallback: Just return text assuming it might be useful content
            return generated_text, {}
