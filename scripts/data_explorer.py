from neo4j import GraphDatabase
import os

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(uri, auth=(user, password))

def find_paths_to_job(driver, job_name):
    query = """
    MATCH path = (s:School)-[:OFFERS|LEADS_TO*..]->(j:Job {name: $job_name})
    RETURN path
    """
    with driver.session() as session:
        result = session.run(query, job_name=job_name)
        paths = [record["path"] for record in result]
        return paths

def print_paths(paths):
    for path in paths:
        nodes = path.nodes
        relationships = path.relationships
        
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            relationship = relationships[i]
            end_node = nodes[i + 1]
            
            print(f"{start_node['name']} -[{relationship.type}]-> {end_node['name']}")



  
    
job_name = "Actuaire"  # Replace with the job name you are interested in
    
paths = find_paths_to_job(driver, job_name)
    
print(print_paths(paths))


    