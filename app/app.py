from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transformers import pipeline, Conversation, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

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

AVAIL_PT = [
                  "facebook/blenderbot-1B-distill",
                  "facebook/blenderbot-400M-distill",
                  "facebook/blenderbot_small-90M",
                  
                  "PygmalionAI/pygmalion-6b",
                  "PygmalionAI/pygmalion-2.7b",
                  "PygmalionAI/pygmalion-1.3b",
                  "PygmalionAI/pygmalion-350m",
                  
                  "microsoft/GODEL-v1_1-large-seq2seq",
                  "microsoft/GODEL-v1_1-base-seq2seq",
                  "allenai/cosmo-xl"]

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

@app.get("/get_models/", status_code=200, response_model=OutModels)
async def modelrun():
    lis = AVAIL_CHATBOTS
    for k in AVAIL_MODELS:
        lis.append(k)
    for k in AVAIL_PT:
        lis.append(k)
    return {'models': lis}

@app.get("/init_models/", status_code=200)
async def modelrun():
    init_models()
    return 200



# ------------------------- MODELS ----------------------------------


def run_model(model, context, query):
    if model in AVAIL_MODELS:
        pipe = pipeline("question-answering", model=model, device=0)
        resp = pipe(context=context, question=query)
        return resp
    elif model in AVAIL_CHATBOTS:
        pipe = pipeline("conversational", model=model, device=0)
        resp = pipe([Conversation(f"You are a helpful chatbot. Here's a series of excerpts from Gandhi's memoirs, some in 3rd and some in 1st person:\n\n\n\n {context} \n\n\n\n Now, using this information, correctly and concisely answer the following question: {query}")])
        print(type(resp))
        return resp
    elif model in AVAIL_PT:
        pipe = pipeline("conversational", model=model, device=0, kwargs={'from_pt': True})
        resp = pipe([Conversation(f"You are a helpful chatbot. Here's a series of excerpts from Gandhi's memoirs, some in 3rd and some in 1st person:\n\n\n\n {context} \n\n\n\n Now, using this information, correctly and concisely answer the following question: {query}")])
        print(type(resp))
        return resp
    else:
        return {'error': 'model not found'}

def init_models():

    pipe = []

    lis = AVAIL_CHATBOTS
    for k in AVAIL_MODELS:
        lis.append(k)
    for k in AVAIL_PT:
        lis.append(k)

    for model in lis:
        if model in AVAIL_MODELS:
            pipe.append(pipeline("question-answering", model=model, device=0))
        elif model in AVAIL_CHATBOTS:
            pipe.append(pipeline("conversational", model=model, device=0))
        elif model in AVAIL_PT:
            if model.startswith("facebook") or model.startswith("microsoft") or model.startswith("allenai"):
                tokenizer = AutoTokenizer.from_pretrained(model)
                model2 = AutoModelForSeq2SeqLM.from_pretrained(model, from_pt=True)
            elif model.startswith("Pygmalion"):
                tokenizer = AutoTokenizer.from_pretrained(model)
                model2 = AutoModelForCausalLM.from_pretrained(model, from_pt=True)