from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "✅ FastAPI is working without LLM or Chroma."}

@app.get("/ping")
def ping():
    return {"pong": True}
