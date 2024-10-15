import os
import spacy
from neo4j import GraphDatabase
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load environment variables (NEO4J credentials)
load_dotenv()

# Alternatively, you can hardcode the Neo4j credentials like this:
NEO4J_URI = "neo4j+s://35dfdea4.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "KDYeM8NVhvtTEoIEut_Cvuc68oL778vzAvnlT028Xb0"

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model for similarity search
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load intent classifier
intent_classifier = pipeline("text-classification", model="distilbert-base-uncased")

# Connect to Neo4j
def connect_to_neo4j():
    # Using hardcoded credentials instead of environment variables
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return driver

# Function to extract entities using spaCy NER
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = [ent.text for ent in doc.ents]
    return entities

# Function to classify intent using HuggingFace pipeline
def classify_intent(user_input):
    result = intent_classifier(user_input)
    return result[0]["label"]

# Function to compute similarity between input and stored subthemes using embeddings
def get_similar_subtheme(user_input, stored_subthemes):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    subtheme_embeddings = model.encode(stored_subthemes, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, subtheme_embeddings)
    best_match_index = cosine_scores.argmax().item()
    return stored_subthemes[best_match_index]

# Function to query Neo4j for response based on a subtheme
def query_neo4j(subtheme):
    query = """
    MATCH (s:Subtheme {name: $subtheme})-[:HAS_RESPONSE]->(r:Response)
    RETURN r.text LIMIT 1
    """
    driver = connect_to_neo4j()
    with driver.session() as session:
        result = session.run(query, subtheme=subtheme)
        record = result.single()
        return record["r.text"] if record else "No response found for this subtheme."

# Main chatbot response logic
def process_chat(user_input, stored_subthemes):
    # Extract entities using NER
    entities = extract_entities(user_input)
    
    # Classify the intent of the user input
    intent = classify_intent(user_input)
    
    # Get the most similar subtheme using embeddings
    best_subtheme = get_similar_subtheme(user_input, stored_subthemes)
    
    # If the intent is related to help, guide accordingly
    if intent == "help":
        return "Let me guide you through the help screens..."

    # Query Neo4j based on the identified subtheme
    response = query_neo4j(best_subtheme)
    
    return response

if __name__ == '__main__':
    # Define your stored subthemes for similarity matching
    stored_subthemes = [
        "Age and Sex",
        "Region",
        "Service Layers",
        "How results are calculated",
        "SA2",
        "Metadata document",
        # Add more subthemes that you want to match against
    ]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        
        # Process the chat with NER, intent classification, and Neo4j query
        response = process_chat(user_input, stored_subthemes)
        print("Assistant:", response)
