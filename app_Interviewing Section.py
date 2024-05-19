import streamlit as st
import os
import tempfile
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import nest_asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.memory import ChatMessageHistory



load_dotenv()


prompt_demo = """
"Comporte-toi comme un consultant en orientation académique. Ton rôle est d'aider un 
étudiant à s'orienter en posant une série de questions une par une, selon le questionnaire ci dessous. Après avoir posé chaque question, attends la réponse de l'étudiant avant de poser la 
suivante." 


Pose-moi la question une par une et je vais te repondre 

et apres mes reponses tu dois les annalyser et me proposer des metiers qui me conviennent le plus

tu peux poser d'autres questions si tu veux 

#### Questionnaire d'Orientation Académique pour les Élèves au Niveau Baccalauréat au 
Maroc 
**Section 1: Intérêts Académiques et Extracurriculaires** 
1. **Quels sujets scolaires trouvez-vous les plus intéressants ?** 
 - a) Mathématiques 
 - b) Sciences physiques et naturelles 
 - c) Littérature et langues 
 - d) Histoire et géographie 
 - e) Arts et musique 
2. **Quelles activités extracurriculaires préférez-vous ?** 
 - a) Sports 
 - b) Musique et arts 
 - c) Club scientifique 
 - d) Club de lecture et écriture 
 - e) Volontariat et actions sociales 
**Section 2: Compétences et Aptitudes** 
3. **Dans quelles matières obtenez-vous les meilleurs résultats ?** 
 - a) Mathématiques 
 - b) Physique/Chimie 
 - c) Français/Anglais 
 - d) Histoire/Géographie 
 - e) Éducation artistique 
4. **Comment décririez-vous votre style d'apprentissage ?** 
 - a) Visuel (préférant des images et des graphiques) 
 - b) Auditif (apprenant mieux en écoutant) 
 - c) Kinesthésique (apprenant en faisant) 
 - d) Lecture/Écriture (préférant lire et écrire) 
 - e) Mixte 
5. **Quelle est votre plus grande force en classe ?** 
 - a) Résolution de problèmes 
 - b) Créativité 
 - c) Communication orale 
 - d) Travail en équipe 
 - e) Recherche et analyse 
**Section 3: Aspirations Professionnelles** 
6. **Quel domaine professionnel vous attire le plus ?** 
 - a) Sciences et ingénierie 
 - b) Santé et médecine 
 - c) Enseignement et éducation 
 - d) Arts et design 
 - e) AƯaires et gestion
 
7. **Quel type d'environnement de travail préférez-vous ?** 
 - a) Bureau structuré 
 - b) Travail en extérieur 
 - c) Laboratoire de recherche 
 - d) Studio artistique 
 - e) Travail à domicile 
**Section 4: Valeurs et Motivations** 
8. **Quelles valeurs sont les plus importantes pour vous dans une carrière ?** 
 - a) Éthique et intégrité 
 - b) Innovation et créativité 
 - c) Aide et soutien aux autres 
 - d) Reconnaissance et prestige 
 - e) Indépendance et autonomie 
**Section 5: Personnalité et Style de Vie** 
9. **Comment préférez-vous résoudre des problèmes ?** 
 - a) En analysant des données et des faits 
 - b) En imaginant des solutions créatives 
 - c) En collaborant avec d'autres 
 - d) En expérimentant diƯérentes approches
 - e) En suivant des méthodes établies 
10. **Comment gérez-vous le stress et la pression ?** 
 - a) En planifiant et en organisant 
 - b) En faisant des activités créatives 
 - c) En parlant avec des amis ou des mentors 
 - d) En faisant du sport ou de l'exercice 
 - e) En prenant des pauses et en me relaxant 




 when you finish the conversation, you have to give a summary of the interests of the student so that we can use it to continue the second part of the conversation

 """




def initialize_session_state():
    if "history" not in st.session_state:
        history= ChatMessageHistory()
        history.add_user_message(prompt_demo)
        st.session_state["history"] = history
        

    if "generated" not in st.session_state:
        st.session_state["generated"] = [
            "Hello! Feel free to ask me any questions about your career interests."
        ]

    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey! 👋"]


def conversation_chat(query, chain, history):
    history.add_user_message(query)
    result = chain.invoke({"messages": history.messages})
    history.add_ai_message(result.content)
    return result.content


def display_chat_history(chain):

    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input = st.text_input(
                "Question:",
                placeholder="I'm here to help you with your career interests.",
                key="messages"
            )
            submit_button = st.form_submit_button(label="Send")

        if submit_button and user_input:
            with st.spinner("Generating response ......"):
                output = conversation_chat(
                    query=user_input,
                    chain=chain,
                    history=st.session_state["history"]
                )

            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)

    if st.session_state["generated"]:
        with reply_container:
            for i in range(len(st.session_state["generated"])):
                message(
                    st.session_state["past"][i],
                    is_user=True,
                    key=str(i) + "_user",
                    avatar_style="thumbs"
                )
                message(
                    st.session_state["generated"][i],
                    key=str(i),
                    avatar_style="fun-emoji"
                )



def create_conversational_chain():
    # llm = ChatAnthropic(model="claude-3-sonnet-20240229")

    llm = AzureChatOpenAI(
        azure_endpoint = "https://gpteel.openai.azure.com/", 
        model="ideta-gpt-3-5-turbo",
        api_version="2024-02-15-preview"
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a career counselor helping a high school moroccan student navigate their academic interests and career aspirations. Your role is to ask a series of questions one by one and only one by one, according to the questionnaire below. After asking each question, wait for the student's response before asking the next one.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

    chain = prompt | llm
    

    return chain






def main():
    initialize_session_state()
    st.title("OrienTech Chatbot")
    st.sidebar.title("Document Processing")


    chain = create_conversational_chain()
    print("********************************")

    display_chat_history(chain)



if __name__ == "__main__":

    main()