from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
import streamlit as st
from langchain.chat_models import ChatOllama, ChatOpenAI


def main():
    load_dotenv()

    st.set_page_config(page_title="Langchain CSV Agent")
    st.header("Langchain CSV Agent📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        agent = create_csv_agent(
            OpenAI(temperature=0),
            #ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
            #ChatOllama(temperature=0, model = "ollama/mistral:latest"),
            csv_file, 
            max_iterations=4,
            verbose=True, 
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()
