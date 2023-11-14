from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.language_models import TextGenerationModel as TGM
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, TFAutoModelForSeq2SeqLM, TFAutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-6B-200K", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-6B-200K", trust_remote_code=True, device_map="auto", torch_dtype="auto")

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

class InAPI(BaseModel):
    context: str
    query: str

class InQ(BaseModel):
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

@app.post("/get-keywords/", status_code=200)
async def model_run(request: InQ):

    resp = keyword(request.query)
    return {'out': resp}

@app.post("/vertex-ai/", status_code=200)
async def model_run(request: InAPI):

    resp = vertex(request.query, request.context)
    return {'out': resp}



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

def keyword(query):

    vertexai.init(project="neon-webbing-404904", location="asia-southeast1")
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 1
    }

    prompt = f'I have a query: {query}\nNow, I need to search for relevant paragraphs in a corpus relating to this query. So, give me just a list of 3 comma-separated keywords (not included in the query) that would be helpful in searching for this query.'

    model = TextGenerationModel.from_pretrained("text-bison")
    response = model.predict(prompt,
        **parameters
    )
    return response.text

def pred_yi(context, query):
    # Use a pipeline as a high-level helper
    

    prompt = f"Background: From the memoirs of Gandhi in both third and first person:\n{context}\n\nQ: {query}\n\nA: "
    inputs = tokenizer(prompt, return_tensors="pt")

    
    outputs = model.generate(
    inputs.input_ids.cuda(),
    max_length=300,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    repetition_penalty=1.3,
    no_repeat_ngram_size=5,
    temperature=0.7,
    top_k=40,
    top_p=0.8,
    )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(out)
    return out.split("A: ")[-1]

def vertex(query, context):

    vertexai.init(project="neon-webbing-404904", location="asia-southeast1")
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 1
    }

    prompt = f'Background: From the memoirs of Gandhi in both third and first person:\n{context}\n\nQ: {query}\n\nA: '

    model = TGM.from_pretrained("text-bison-32k")
    response = model.predict(prompt,
        **parameters
    )
    return response.text