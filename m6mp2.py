import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# STEP 1 : Load document
loader  = CSVLoader(file_path='./dataset/titanic.csv')
data    = loader.load()
print(data)

# STEP 2 : Split data into chunks
text_splitter   = RecursiveCharacterTextSplitter(
    chunk_size  = 500,
    chunk_overlap   = 50
)

splits  = text_splitter.split_documents(data)
print(len(splits))

# Step 3: Create embeddings,  store in Vector Database 
# Step 4: Perform the retrieval augmented generation by integrating with LLM 
# Step 5: Frame at least five logical questions relevant to the knowledge base and demonstrate relevant answers from the RAG system 
# Step 6: Create a Gradio App where user can write the query and get the response from the RAG system
# Step 7: Deploy the application on HF Spaces



### DELETE
# file    = ('dataset/titanic.csv')
# data    = pd.read_csv(file) 
# print(data.shape)
