from langchain_groq import ChatGroq
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use ["http://localhost:5173"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




groq_api_key = "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"

llm = ChatGroq(model="Llama3-70b-8192", groq_api_key=groq_api_key, temperature=0.5)

# Define a Pydantic model for the request body
class MessageRequest(BaseModel):
    message: str

@app.post("/getresponse")
def get_response(request: MessageRequest):
    print("Message: ", request.message)
    return {"response": llm.invoke(request.message)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
