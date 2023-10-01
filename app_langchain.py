import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from itertools import zip_longest
from streamlit_chat import message

from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.7)

st.set_page_config(page_title="Sayjini - Your Personal SuperBot", page_icon=":robot_face:")
st.title("RAG based contextual search chatbot")
st.markdown("Jai shree krishna 🙏🏻") 


FILE_PATH = "vector_index_BigBazaar_head_1000.pkl"

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

if 'sources' not in st.session_state:
    st.session_state['sources'] = []  # Store the sources
    

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="""
        you will first translate the input query into english.
        you will extract data from the translated query in the format of json schema provided.
        json schema:  {
                "query": {
                    "items": {
                    "brand": "string",
                    "quantity": "number",
                    "keywords": ["string"],
                    "price_range": {
                        "minimum_price": "number",
                        "maximum_price": "number"
                    }
                    },
                    "filters": {
                    "include": ["string"],
                    "exclude": ["string"],
                    "category_or_type": ["string"]
                    }
                }
            }
        """
)]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages

def generate_response():
    """
    
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    return ai_response.content


def create_structured_output(message):
    json_schema = {
        "type": "object",
        "properties": {
            "order": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "item": {
                    "type": "string"
                },
                "quantity": {
                    "type": "number"
                },
                "price": {
                    "type": "number"
                }
                },
                "required": ["item", "price"]
            }
            },
            "total": {
            "type": "number"
            }
        },
        "required": ["order", "total"]
        }
    prompt_msgs = [
    SystemMessage(
        content=f"You are a world class algorithm for extracting information in structured formats."
    ),
    HumanMessage(content="Use the given format to extract information from the following input:"),
    HumanMessagePromptTemplate.from_template("{input}"),
    HumanMessage(content="Tips: Make sure to answer in the correct format"),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)
    chain = create_structured_output_chain(json_schema, chat, prompt , verbose=True)
    return chain.run(message)


def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)
    
    if user_query:
        if os.path.exists(FILE_PATH):
            with open(FILE_PATH, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=chat, retriever=vectorstore.as_retriever())
                result = chain({"question": user_query}, return_only_outputs=True)
                # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                # st.header("Answer")
                # st.write(result["answer"])

                # Display sources, if available
                answer_source = ""
                sources = result.get("sources", "")
                if sources:
                    # st.subheader("Sources:")
                    sources_list = sources.split("\n")  # Split the sources by newline
                    for source in sources_list:
                        # st.write(source)
                        answer_source += source + "\n"
    # Generate response
    output = result["answer"]

    # Append AI response to generated responses
    st.session_state.generated.append(output)
    # Append sources to sources
    st.session_state.sources.append(answer_source)
    # Clear prompt_input
    st.session_state.prompt_input = ""

st.text_input('YOU: ', key='prompt_input', on_change=submit)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i),is_user=False,avatar_style="miniavs")
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user',avatar_style="adventurer")
        
 #+ "\n "+ "Sources \n"+ st.session_state["sources"][i]       
# Add credit
st.markdown("""
---
Made with 🤖 by DataWithD🥂""")