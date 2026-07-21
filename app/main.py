from fastapi import FastAPI
import routes

app = FastAPI(title="RAG API", description="API for RAG operations", version="1.0.0")

app.include_router(routes.router)
