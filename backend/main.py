from fastapi import FastAPI
from backend.api import upload, train, predict, models

app = FastAPI()

app.include_router(upload.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(models.router)