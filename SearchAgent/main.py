from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()


def main():
    tools = [TavilySearch()]
    # llm = ChatOllama(temprature=0, model="llama3.2:latest")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(AgentResponse)
    # output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
    # react_prompt = hub.pull("hwchase17/react")
    react_prompt = PromptTemplate(
        template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
        input_variables=["input", "agent_scratchpad", "tool_names"]
    ).partial(
        format_instructions=""
    )
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )
    
    extract_output = RunnableLambda(lambda x: x["output"])
    # parse_output = RunnableLambda(lambda x: output_parser.parse(x))
    chain = agent_executor | extract_output | structured_llm

    result = chain.invoke(
        input={
            "input": "search for 3 job postings for a software engineer in Dhaka, Bangladesh on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
