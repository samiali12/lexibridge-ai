from src.data_processor import DataProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.llm import LLM
import streamlit as st 
import re 
import json 
from deep_translator import GoogleTranslator



@st.cache_resource
def load_pipeline():
    data_obj = DataProcessor(limit=5)
    chunks, _ = data_obj.build_data()

    embedding = EmbeddingManager()
    model = embedding.get_model()

    chunks_list = [c.page_content for c in chunks]
    embd = embedding.embed_texts(chunks_list)

    vectordb = VectorStore()
    vectordb.add_document(chunks, embd)
    retriever = vectordb.get_retriever(model)

    llm = LLM(retriever)
    return llm

def highlight_text(query, text):
    words = set(query.split()) 
    for word in words:
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

def translate(text, target_lang='ur'):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

if __name__ == '__main__':
    
    llm = load_pipeline()

    st.set_page_config(page_title="Legal RAG System", layout="wide")

    st.title("Lexi bridge AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # sidebar 

    with st.sidebar:
        st.header("⚙️ Settings")

        selected_language = st.selectbox(
            "Select Language",
            ("English", "Urdu"),
        )

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
        
         with st.chat_message("user"):
            st.markdown(chat['query'])
         with st.chat_message('assistant'):
             st.markdown(chat['answer'])

         with st.expander("📚 Sources"):
             for i, doc in enumerate(chat["sources"], 1):
                highlighted = highlight_text(doc['page_content'][:300], chat['query'])
                st.markdown(highlighted, unsafe_allow_html=True)
                st.caption(f"**Source:** {doc['metadata']['source']} | Length: {doc['metadata']['content_length']}")

    if query := st.chat_input("Enter your legal question..."):
    
        with st.spinner("🤖 Thinking... please wait"):
                result = llm.invoke(query)

        if selected_language.lower() == 'urdu':
            with st.spinner('Translating into urdu'):
                st.session_state.chat_history.append({
                    "query": translate(query),
                    "answer": translate(result['result']),
                    "sources": [
                        {
                            "page_content": translate(doc.page_content),
                            "metadata": doc.metadata
                        }
                        for doc in result["source_documents"]
                    ]
                })
        else:
            st.session_state.chat_history.append({
                    "query": query,
                    "answer": result['result'],
                    "sources": [
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in result["source_documents"]
                    ]
            })

        st.rerun()


            

    