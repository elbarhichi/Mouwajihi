import streamlit as st
import os
import dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory



dotenv.load_dotenv()

prompt_demo = """
"Comporte-toi comme un consultant en orientation académique. Ton rôle est d'aider un 
étudiant à s'orienter en posant une série de questions une par une, selon le questionnaire cidessous. Après avoir posé chaque question, attends la réponse de l'étudiant avant de poser la 
suivante." 


Pose-moi la question une par une en me donnat les choix et je vais te repondre 

et apres mes reponses tu dois les annalyser et me proposer des metiers qui me conviennent le plus


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

 
"""

llm = AzureChatOpenAI(
  azure_endpoint = os.getenv("AZURE_ENDPOINT"), 
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

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(prompt_demo)

st.title('OrienTech Chatbot')

def generate_response(input_text):
    
    response = chain.invoke({"messages": demo_ephemeral_chat_history.messages})
    demo_ephemeral_chat_history.add_ai_message(response.content)

    st.info(response)

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
  submitted = st.form_submit_button('Submit')
  if submitted and text:
    generate_response(text)