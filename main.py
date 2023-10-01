import pandas as pd
import numpy as np
import os
import openai
from dotenv import load_dotenv
import ast
import faiss
import json

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key

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
from langchain.chat_models import ChatOpenAI

from openai.embeddings_utils import get_embedding
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000 

chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k",temperature=0)

st.set_page_config(page_title="Sayjini - Your Personal SuperBot", page_icon=":robot_face:")
st.title("RAG based contextual search chatbot")
st.markdown("Jai shree krishna 🙏🏻") 

# FILE_PATH_CSV = st.file_uploader("Upload CSV database file", type="csv")
FILE_PATH_CSV = "git_data/BigBazaar_vector_db_git.csv"
# FILE_PATH_INDEX = st.file_uploader("Upload vector index in pickle format", type="pkl")
FILE_PATH_INDEX = "git_data/vector_index.pkl"

if FILE_PATH_CSV is not None and FILE_PATH_INDEX is not None:
    # Initialize session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []  # Store AI generated responses

    if 'past' not in st.session_state:
        st.session_state['past'] = []  # Store past user inputs

    if 'entered_prompt' not in st.session_state:
        st.session_state['entered_prompt'] = ""  # Store the latest user input

    if 'sources' not in st.session_state:
        st.session_state['sources'] = []  # Store the sources
        
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = []  # Store the sources
        
    if 'recommended_items' not in st.session_state:
        st.session_state['recommended_items'] = []  # Store the recommended items
        
    if 'user_queries' not in st.session_state:
        st.session_state['user_queries'] = ""   # Store the user queries

        

    def build_message_list():
        """
        Build a list of messages including system, human and AI messages.
        """
        # Start zipped_messages with the SystemMessage
        zipped_messages = [SystemMessage(
            content="""
            you have to filter the items in top_k_results based on the filters provided below and show only the filtered results to the customer.
            you have to respond all the items provided below to a customer briefly like a shopkeeper.
            Only show the items which are according to the filters provided below.
            """ )]

        # Zip together the past and generated messages
        for human_msg,recommended_items, ai_msg in zip_longest(st.session_state['past'],st.session_state['recommended_items'], st.session_state['generated']):
            if human_msg is not None:
                zipped_messages.append(HumanMessage(
                    content=human_msg))  # Add user messages
            if recommended_items is not None:
                zipped_messages.append(
                    AIMessage(content=recommended_items)) # Add recommended items
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

    def create_vector_index(path_to_csv):
        df = pd.read_csv(path_to_csv)
        # Apply ast.literal_eval to convert the strings back to lists
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        vectors = df.embedding.to_list()
        vectors = np.array(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index,df

    def load_vector_index(path_to_csv,path_to_index):
        print(path_to_csv,path_to_index)
        df = pd.read_csv(path_to_csv)
        index = faiss.read_index(path_to_index)
        return index,df
        

    if FILE_PATH_INDEX is None:
        index,df = create_vector_index(FILE_PATH_CSV)
    else:
        index,df = load_vector_index(FILE_PATH_CSV,FILE_PATH_INDEX)

    def process_search_query(query,df,index):
        search_vec = get_embedding(query, engine=embedding_model)
        search_vec = np.array(search_vec).reshape(1, -1)
        distances, I = index.search(search_vec, k=2)
        I.tolist()
        row_indices = I.tolist()[0]
        top_k_results = df[['Name', 'Brand', 'Price', 'DiscountedPrice','Category', 'SubCategory', 'Quantity']].loc[row_indices]
        return top_k_results.to_dict(orient='records')

    def get_completion(prompt, model="gpt-3.5-turbo-16k"):
        messages = [{"role": "system", "content": """
            you will first translate the input query into english and for each identified item, write the query for searching in vector database of products
            you will extract data from the translated query in the format of json schema provided.
            json schema:  {
                    "query": {
                        "tranlsated_query": "string",
                        "items": [{
                        "item_name": "string",
                        "search_query": "string",
                        "brand": "string",
                        "quantity": "number",
                        "unit": "string",
                        "Important_words": ["string"],
                        "features": ["string"],
                        "occasion": ["string"],
                        "price_range": {
                            "minimum_price": "number",
                            "maximum_price": "number"
                        }
                    }]
                }
            """},
            {"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]

    def submit():
        # Set entered_prompt to the current value of prompt_input
        st.session_state.entered_prompt = st.session_state.prompt_input
        # Get user query
        user_query = st.session_state.entered_prompt
        # Append user query to past queries
        st.session_state.past.append(user_query)
        # Generate response
        st.session_state['user_queries'] += user_query + " || "
        print(st.session_state['user_queries'])
        output = get_completion(user_query)
        # output = get_completion(st.session_state['user_queries'])
        print(output)
        try:
            output = json.loads(output)
            response = ""
            recommendations = []
            for item in range(len(output["query"]["items"])):
                # print(output["query"]["items"][item])
                top_k_results = process_search_query(str(output["query"]["items"][item]),df,index)
                recommendations.append(top_k_results)
                response += str(top_k_results) + "\n"  
            st.session_state['recommendations'] = recommendations
            response ="top_k_results: \n" + response+ "Filters: \n" + str(output) 
        except:
            response = str(output)
        st.session_state['recommended_items'].append(response)
        generated_response =  generate_response()
        
        # Append AI response to generated responses
        st.session_state.generated.append(generated_response)
        # st.session_state['user_queries'] += generated_response + "\n"
        # Append sources to source
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

    with st.sidebar:
        st.write(st.session_state['recommendations'])      
 #+ "\n "+ "Sources \n"+ st.session_state["sources"][i]       
# Add credit
st.markdown("""
---
Made with 🤖 by DataWithD🥂""")

# mujhe mere chote bhai ke birthday ke liye oppo ka phone chaiye, jisme 6 gb ram ho aur 128 gb storage ho, aur uska price 20000 se 30000 ke beech ho 