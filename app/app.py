from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Cached model
cache_path = "./t5"

# Huggingface model
model_id = "t5-large"

AVAIL_MODELS = ["deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
                "bert-large-uncased-whole-word-masking-finetuned-squad",
                "timpal0l/mdeberta-v3-base-squad2",
                "deepset/tinyroberta-squad2",
                "deepset/minilm-uncased-squad2"]

AVAIL_CHATBOTS = [
                  "microsoft/DialoGPT-small",
                  "microsoft/DialoGPT-medium",
                  "microsoft/DialoGPT-large",
                  
                  ]

app = FastAPI(title="COL106 A7 custom API lesssgooooooo")
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InParams(BaseModel):
    model: str
    context: str
    query: str

class OutModels(BaseModel):
    models: List[str]

@app.post("/api/", status_code=200)
async def modelrun(request: InParams):

    response = run_model(request.model, request.context, request.query)

    if not response:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(
            status_code=404, detail="404 error"
        )
    
    return response

@app.post("/yi/", status_code=200)
async def modelrun(request: InParams):

    response = pred_yi(request.context, request.query)

    if not response:
        # the exception is raised, not returned - you will get a validation
        # error otherwise.
        raise HTTPException(
            status_code=404, detail="404 error"
        )
    
    return response

@app.get("/get_models/", status_code=200, response_model=OutModels)
async def modelrun():
    lis = AVAIL_CHATBOTS
    for k in AVAIL_MODELS:
        lis.append(k)
    return {'models': lis}

@app.get("/init_models/", status_code=200)
async def modelrun():
    init_models()
    return 200

@app.post("/t5-inf/", status_code=200)
async def model_run(request: InParams):

    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id, cache_dir=cache_path).to('cuda')

    input = tokenizer(request.query, return_tensors="pt", padding=True).to('cuda')
    generate_text = model.generate(
        input_ids=input["input_ids"],
        attention_mask=input["attention_mask"],
        do_sample=False,
    )
    generated_text = tokenizer.batch_decode(generate_text, skip_special_tokens=True)
    generated_text = generated_text[0]
    print(generated_text)
    return {'out': generated_text}




# ------------------------- MODELS ----------------------------------


def run_model(model, context, query):
    if model in AVAIL_MODELS:
        pipe = pipeline("question-answering", model=model, device=0)
        resp = pipe(context=context, question=query)
        return resp
    elif model in AVAIL_CHATBOTS:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
        model2 = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to('cuda')
        prompt = (f"You are a helpful chatbot. Here's a series of excerpts from Gandhi's memoirs, some in 3rd and some in 1st person:\n\n\n\n {context} \n\n\n\n Now, using this information, correctly and concisely answer the following question: {query}")
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt').to('cuda')
        chat_history_ids = model2.generate(new_user_input_ids, max_length=4000, pad_token_id=tokenizer.eos_token_id)
        print((tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)))
        
        res = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        return {'response': res}
    else:
        return {'error': 'model not found'}

def init_models():

    pipe = []

    lis = AVAIL_CHATBOTS
    for k in AVAIL_MODELS:
        lis.append(k)

    for model in lis:
        if model in AVAIL_MODELS:
            pipe.append(pipeline("question-answering", model=model, device=0))
        elif model in AVAIL_CHATBOTS:
            pipe.append(pipeline("conversational", model=model, device=0))

def pred_yi(context, query):
    # Use a pipeline as a high-level helper
    from transformers import pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B")
    model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B", trust_remote_code=True, device_map="auto", torch_dtype="auto")

    prompt = f"Background: From the memoirs of Gandhi in both third and first person:\n{context}\n\nQ: {query}\n\nA: "
    inputs = tokenizer(prompt, return_tensors="pt").input_ids

    pred = model.generate(inputs, max_new_tokens=300, do_sample=True, top_k=50, top_p=0.95)
    out = tokenizer.batch_decode(pred, skip_special_tokens=True)[0]
    print(out)
    return out