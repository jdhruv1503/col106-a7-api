from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline, Conversation

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
                  "JosephusCheung/Guanaco",
                  "TheBloke/Vicuna-13B-1.1-GPTQ",
                  "alpindale/goliath-120b",
                  "h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
                  "facebook/blenderbot-3B",
                  "facebook/blenderbot-1B-distill",
                  "facebook/blenderbot-400M-distill",
                  "facebook/blenderbot_small-90M",
                  "PygmalionAI/pygmalion-6b",
                  "PygmalionAI/pygmalion-2.7b",
                  "PygmalionAI/pygmalion-1.3b",
                  "PygmalionAI/pygmalion-350m",
                  "microsoft/GODEL-v1_1-large-seq2seq",
                  "microsoft/GODEL-v1_1-base-seq2seq",
                  "allenai/cosmo-xl"
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
    models: list

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

@app.get("/get_models/", status_code=200)
async def modelrun():
    return {'models': AVAIL_MODELS.extend(AVAIL_CHATBOTS)}



# ------------------------- MODELS ----------------------------------


def run_model(model, context, query):
    if model in AVAIL_MODELS:
        pipe = pipeline("question-answering", model=model, device=0)
        resp = pipe(context=context, question=query)
        return resp
    elif model in AVAIL_CHATBOTS:
        pipe = pipeline("conversational", model=model, device=0)
        resp = pipe(Conversation(f"You are a helpful chatbot. Here's a series of excerpts from Gandhi's memoirs, some in 3rd and some in 1st person:\n\n\n\n {context} \n\n\n\n Now, using this information, correctly and concisely answer the following question: {query}"))
        return resp
    else:
        return {'error': 'model not found'}