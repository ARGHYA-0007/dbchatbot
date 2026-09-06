from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dbchatbot_model import workflow
from langchain_core.messages import HumanMessage
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "this is our database"}


@app.post("/predict")
def predict(query: str):

    result = workflow.invoke(
        {
            "messages": [
                HumanMessage(content=query)
            ]
        }
    )

    # Check all messages produced by LangGraph
    for message in result["messages"]:

        if getattr(message, "name", None) in ("get_students", "query_students"):

            students = json.loads(message.content)

            if not students:
                return {
                    "type": "text",
                    "data": "No students found in the database."
                }

            return {
                "type": "table",
                "data": students
            }

    # If no database table was requested,
    # return the normal AI response
    final_message = result["messages"][-1]

    return {
        "type": "text",
        "data": final_message.content
    }