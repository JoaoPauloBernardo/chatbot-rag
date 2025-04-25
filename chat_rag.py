from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain.prompts import PromptTemplate
import json



# Carrega configurações
with open('config.json') as f:
    config = json.load(f)

# Template otimizado
template = """
Você é um professor Pokémaster. Responda com dados EXATOS da Pokédex:

Contexto:
{context}

Pergunta: {question}
Resposta concisa (máx 2 frases):"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Configuração do modelo
tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=pipe)
embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model"])
vectordb = Chroma(persist_directory="db", embedding_function=embeddings)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    verbose=False
)

# Interface otimizada
def chat_pokemon():
    print("Professor Carvalho: Bem-vindo à Pokédex Interativa!")
    while True:
        query = input("\nVocê: ")
        if query.lower() in ["sair", "exit"]:
            break
        
        try:
            result = qa({"question": query, "chat_history": []})
            print(f"\nPokédex: {result['answer'].split('Resposta:')[-1].strip()}")
        except Exception as e:
            print(f"\nErro: {e}")

if __name__ == "__main__":
    chat_pokemon()