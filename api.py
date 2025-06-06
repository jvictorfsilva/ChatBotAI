import logging
import json
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import config

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI()

model_path = str(config.MODEL_OUTPUT_DIR)

logger.info(f"Loading model from {model_path}")
model = AutoModelForCausalLM.from_pretrained(model_path).to(config.DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_path)
logger.info("Model and tokenizer loaded successfully.")


class UserInput(BaseModel):
    prompt: str


def log_interaction(prompt: str, response: str):
    log_entry = {"prompt": prompt, "response": response}
    with open(config.CONVERSATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


@app.post("/chat")
async def chat(user_input: UserInput):
    prompt = user_input.prompt
    logger.info(f"Received prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.MAX_LENGTH,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

    response_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    logger.info(f"Generated response: {response}")

    log_interaction(prompt, response)

    return {"response": response}


@app.get("/")
def read_root():
    return {"message": "AI Chatbot API is running. Send a POST request to /chat"}


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
