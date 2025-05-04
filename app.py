from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
LLMapp = workflow.compile(checkpointer=memory)



app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_model(request: Request):
    config = {"configurable": {"thread_id": "abc123"}}
    client_ip = request.client.host
    config["configurable"]["thread_id"] = client_ip
    user_agent = request.headers.get("user-agent", "").lower()
    if "mobile" in user_agent:
        config["configurable"]["thread_id"] = f"{client_ip}_mobile"
    data = await request.json()
    query = data.get("query")
    input_messages = [HumanMessage(query)]
    output = LLMapp.invoke({"messages": input_messages}, config)
    response_message = output["messages"][-1].content
    return {"response": response_message}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)