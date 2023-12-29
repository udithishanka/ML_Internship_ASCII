import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css

from func import handle_userinput, get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain

def main():
    #Loading API Keys
    load_dotenv()

    #Page Formatting
    st.set_page_config(page_title="Chat with PDFs")
    st.write(css, unsafe_allow_html=True)
    st.header("PDF Chatbot")
    user_question = st.text_input("Ask a question about your documents:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state: 
        st.session_state.chat_history = None

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", type='pdf')
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                store_name = pdf_docs.name[:-4]
                st.session_state.vectorstore = get_vectorstore(text_chunks,store_name)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)


if __name__ == '__main__':
    main()
