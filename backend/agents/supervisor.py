import os

from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from agents.prompts import SUPERVISOR_PROMPT

from tools.dataset_tool import dataset_tool
from tools.preprocessing_tool import preprocessing_tool
from tools.training_tool import training_tool
from tools.evaluation_tool import evaluation_tool
from tools.explainability_tool import explainability_tool
from tools.report_tool import report_tool


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [
    dataset_tool,
    preprocessing_tool,
    training_tool,
    evaluation_tool,
    explainability_tool,
    report_tool,
]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SUPERVISOR_PROMPT,
)