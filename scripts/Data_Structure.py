from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import json
import google.generativeai as genai
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Configure the AI API
genai.configure(api_key=os.getenv("GEmin_KEY"))

# Function to extract and structure information from a PDF page
def extract_information_from_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = ""
    
    # Concatenate text from each page
    for page in reader.pages:
        full_text += page.extract_text() + "\n"  # Ensure separation between pages

    # Define the extraction prompt more specific to educational programs and their connections
    prompt = """
        Analyze the provided text about educational institutions and their programs. Identify and structure
        information into nodes representing Schools, Filières, and Master's Specializations, along with their 
        properties such as website, name, full name, and location. Also, extract potential job outcomes. Format
        the extracted data in a JSON schema that reflects entities and relationships for a NEO4J graph database, 
        considering 'offers' and 'leads_to' relationships. Example output should include:
        {
            "Schools": [
                {"name": "INSEA", "website": "https://insea.ac.ma/", "fullName": "Institut National de Statistique et d'Économie Appliquée", "location": "Rabat"},
                ...
            ],
            "Filières": [
                {"name": "Actuariat – Finance", "offeredBy": "INSEA"},
                ...
            ],
            "MastersSpecialises": [
                {"name": "Comptabilité, Contrôle de Gestion et Audit (CCA)", "offeredBy": "ENCG Casablanca"},
                ...
            ],
            "Jobs": [
                {"name": "Actuaire", "leadsTo": ["Actuariat – Finance"]},
                ...
            ]
        }
        Ensure that the relationships are accurately mapped according to the 'offers' and 'leads_to' specifications.
    """

    # Call the generative AI model
    model = genai.GenerativeModel('gemini-pro')
    generation_config = genai.GenerationConfig(temperature=0.5)
    response = model.generate_content([full_text, prompt], generation_config=generation_config)

    return json.loads(response.text)

# Extract information from a specific PDF
extracted_data = extract_information_from_pdf('path_to_your_pdf.pdf')

# Save the extracted information to a new JSON file
with open('structured_data_for_neo4j.json', 'w') as output_file:
    json.dump(extracted_data, output_file, indent=4)
