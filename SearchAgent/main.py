from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello, Langchain!")
    tools = [TavilySearch()]
    # llm = ChatOllama(temprature=0, model="gemma3:270m")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
        )
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    chain = agent_executor

    result = chain.invoke(
        input={
            "input": "search for 3 job postings for a software engineer in Dhaka, Bangladesh on linkedin and list their details",
        }
    )
    print(result)

if __name__ == "__main__":
    main()