# 🤖 Chatbot Pokémon (Geração 1)

Este projeto é um chatbot inteligente que responde perguntas sobre os Pokémon da Primeira Geração utilizando:

- 🧠 Recuperação aumentada de conhecimento (RAG) com LangChain + ChromaDB
- 🌐 Integração com PokéAPI em tempo real
- ⚡ Cache local via JSON para performance e fallback
- 🖥️ Interface moderna usando Gradio

---

## 🚀 Como rodar o projeto

### 1. Clone o repositório
git clone https://github.com/JoaoPauloBernardo/chatbot-rag.git
cd chatbot-rag

2. Instale as dependências
pip install -r requirements.txt

4. Gere a base vetorizada
python scripts/criar_base_vetorizada.py
(Gera a base a partir do arquivo pokemons_primeira_geracao.pdf)

5. Rode o chatbot
python scripts/chat_rag.py

7. Acesse no navegador
http://127.0.0.1:7860

💬 Exemplos de perguntas que você pode fazer

Qual o tipo do Charizard?

Quais são as habilidades do Pikachu?

Me fale sobre o Gyarados.

Qual a velocidade do Gengar?

Ataque especial do Alakazam?


🧩 Tecnologias utilizadas
Python 3.10

LangChain

ChromaDB

Sentence Transformers

Gradio

PokéAPI

💬 Sobre o projeto
Este chatbot foi desenvolvido como estudo de técnicas modernas de RAG (Retrieval-Augmented Generation) com integração de APIs externas, demonstrando habilidades práticas em Inteligência Artificial Aplicada, Engenharia de Dados e Desenvolvimento Web com Python.

🏆 Autor
João Paulo Bernardo
https://www.linkedin.com/in/jo%C3%A3o-paulo-fonseca-bernardo-169971246/
