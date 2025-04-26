# 🤖 Chatbot RAG Pokémon (Geração 1)

Este é um chatbot baseado em RAG (Retrieval-Augmented Generation) que responde perguntas sobre Pokémon da 1ª geração utilizando:

- 🧠 **LangChain** para recuperação de contexto.
- 🗃️ **ChromaDB** para base vetorizada.
- 📄 **PDF com informações dos Pokémon**.
- 🌐 **PokéAPI** para complementar com dados atualizados.

## 🚀 Como rodar

```bash
pip install -r requirements.txt
python criar_base_vetorizada.py
python interface.py
