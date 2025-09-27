from src.data_processor import DataProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.llm import LLM
import streamlit as st 
import re 
import json 


@st.cache_resource
def load_pipeline():
    data_obj = DataProcessor(limit=5)
    chunks, data = data_obj.build_data()

    embedding = EmbeddingManager()
    model = embedding.get_model()

    chunks_list = [c.page_content for c in chunks]
    embd = embedding.embed_texts(chunks_list)

    vectordb = VectorStore()
    vectordb.add_document(data, embd)
    retriever = vectordb.get_retriever(model)

    llm = LLM(retriever)
    return llm

def highlight_text(query, text):
    for word in query.split():
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"<mark>{word}</mark>", text)
    return text

def get_chat_transcript_text():
    transcript = ""
    if "chat_history" in st.session_state and st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            query = chat['query']
            answer = chat['answer']
            transcript += f"🧑 You: {query}\n🤖 Assistant: {answer}\n\n"
    return transcript if transcript else "No conversation yet."

def get_chat_transcript_json():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        return json.dumps(st.session_state.chat_history, indent=2, ensure_ascii=False)
    return json.dumps({'data': 'not found'},indent=2)

if __name__ == '__main__':
    
    llm = load_pipeline()

    st.set_page_config(page_title="Legal RAG System", layout="wide")

    st.title("Lexi bridge AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        if st.button('🗑️ Clear Chat'):
            st.session_state.chat_history = []

        st.download_button( 
            label="📥 Download Chat (TXT)",
            data=get_chat_transcript_text(),
            file_name="chat_transcript.txt",
            mime="text/plain" 
        )
        st.download_button(
            label="📥 Download Chat (JSON)",
            data=get_chat_transcript_json(),
            file_name="chat_transcript.json",
            mime="application/json"
        ) 


    st.subheader("💬 Conversation")

    for chat in st.session_state.chat_history:
         st.markdown(f"**🧑 You:** {chat['query']}")
         st.markdown(f"**🤖 Assistant:** {chat['answer']}")

         with st.expander("🔎 Sources"):
             for i, doc in enumerate(chat["sources"], 1):
                highlighted = highlight_text(doc.page_content[:300], chat['query'])
                st.markdown(highlighted, unsafe_allow_html=True)
                st.caption(f"Metadata: {doc.metadata}")

    # sample question: 
    query = st.text_input("Enter your legal question:", key="query_input")

    if st.button("search"):
        if query.strip():
            
            with st.spinner("🤖 Thinking... please wait"):
                result = llm.invoke(query)
                
            
            st.session_state.chat_history.append({
                    "query": query,
                    "answer": result['result'],
                    "sources": result["source_documents"]
            })

            st.rerun()

        else:
            st.warning("Please enter a query.")


            

    