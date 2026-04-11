from fastapi import FastAPI
from pydantic import BaseModel
from agents import ManagerAgent

app = FastAPI(title="多智能体协同办公系统", version="1.0")
manager_agent = ManagerAgent()

class TaskRequest(BaseModel):
    task: str

@app.post("/run_task")
async def run_agent_task(request: TaskRequest):
    result = manager_agent.run(request.task)
    return result
# 
# 启动命令：uvicorn main:app --host 0.0.0.0 --reload