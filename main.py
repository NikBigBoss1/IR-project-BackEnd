from fastapi import FastAPI
from routers import search

app = FastAPI()

# Include the search router
app.include_router(search.router)