from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import uuid

load_dotenv()

# Initialize the model
model = init_chat_model("llama3-8b-8192", model_provider="groq")

# Create LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
LLMapp = workflow.compile(checkpointer=memory)

app = FastAPI()

# Mount static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SHARED_THREAD_ID = "shared-context"
ADMIN_MESSAGES = []  # holds latest admin context messages

@app.get("/", response_class=HTMLResponse)
async def read_user(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def read_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.post("/admin/query")
async def admin_query_model(request: Request):
    global ADMIN_MESSAGES
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            raise ValueError("Missing query")

        config = {"configurable": {"thread_id": SHARED_THREAD_ID}}
        input_messages = [HumanMessage(query)]
        output = LLMapp.invoke({"messages": input_messages}, config)
        ADMIN_MESSAGES = output["messages"]  # cache for users
        response_message = output["messages"][-1].content
        return {"response": response_message}

    except Exception as e:
        print(f"[ADMIN ERROR] {e}")
        return {"response": f"Internal Server Error: {str(e)}"}

@app.post("/query")
async def query_model(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        session_id = data.get("session_id")

        if not query:
            raise ValueError("Missing query")

        user_thread_id = f"user-{session_id or str(uuid.uuid4())}"

        # Combine admin context with user query
        input_messages = ADMIN_MESSAGES + [HumanMessage(query)] if ADMIN_MESSAGES else [HumanMessage(query)]
        config = {"configurable": {"thread_id": user_thread_id}}

        output = LLMapp.invoke({"messages": input_messages}, config)
        response_message = output["messages"][-1].content
        return {"response": response_message}

    except Exception as e:
        print(f"[USER ERROR] {e}")
        return {"response": f"Internal Server Error: {str(e)}"}

@app.post("/admin/reset")
def reset_admin_context():
    global ADMIN_MESSAGES
    ADMIN_MESSAGES = []
    return {"message": "Admin context cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
