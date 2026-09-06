from fastapi import FastAPI
from dbchatbot_model import workflow
from langchain_core.messages import HumanMessage
app = FastAPI()
@app.get('/')
def home():
    return {'message':'this is our database'}
@app.post('/predict')
def predict(query:str):
        result = workflow.invoke(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ]
        }
    )
        final_message = result["messages"][-1]
        return final_message.content