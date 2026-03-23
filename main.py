from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
  return {
    "name": "FactN Assistant Service",
    "status": "running"
  }
