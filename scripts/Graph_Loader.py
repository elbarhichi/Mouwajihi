from neo4j import GraphDatabase
import os
import json


uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")


# Function to load data from JSON and import into Neo4j
def import_data(json_data):
    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        # Cypher query to import data
        query = """
       
        // Import Schools
        UNWIND $data['Schools'] AS school
        MERGE (s:School {name: school.name})
        ON CREATE SET s.website = school.website, s.fullName = school.fullName, s.location = school.location

        // Import Filières and link to Schools
        UNWIND $data['Filières'] AS filiere
        MATCH (sch:School {name: filiere.offeredBy})
        MERGE (f:Filiere {name: filiere.name})
        MERGE (sch)-[:OFFERS]->(f)

        // Import Master's Specialisations and link to Schools
        UNWIND $data['MastersSpecialises'] AS master
        MATCH (sch:School {name: master.offeredBy})
        MERGE (m:MastersSpecialise {name: master.name})
        MERGE (sch)-[:OFFERS]->(m)

        // Import Jobs and link to Filières
        UNWIND $data['Jobs'] AS job
        MERGE (j:Job {name: job.name})
        UNWIND job.leadsTo AS leadToFiliere
        MATCH (f:Filiere {name: leadToFiliere})
        MERGE (f)-[:LEADS_TO]->(j)
        """
        # Execute Cypher query
        session.run(query, data=json_data)

    # Close the Neo4j driver
    driver.close()

# Load JSON data from file
with open("D:\\your_path\\structured_data_for_neo4j.json", "r") as file:
    structured_data = json.load(file)

# Call the import_data function and pass JSON data as a parameter
import_data(structured_data)
        
 
