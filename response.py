import os
import csv
import logging
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from difflib import SequenceMatcher  # Import SequenceMatcher for string similarity matching

# Load environment variables
load_dotenv()

# Neo4j Connection
def connect_to_neo4j():
    uri = os.getenv("NEO4J_URI", "neo4j+s://35dfdea4.databases.neo4j.io")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "KDYeM8NVhvtTEoIEut_Cvuc68oL778vzAvnlT028Xb0")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Extract keywords from subtheme stored in Neo4j
def extract_keywords_from_subtheme(subtheme):
    return [keyword.strip().lower() for keyword in subtheme.split(",")]

# Improved keyword matching logic
def match_user_input(user_input, stored_data):
    user_input = user_input.strip().lower()
    
    best_match = None
    highest_similarity = 0
    
    # Tokenize user input into individual words (keywords)
    user_keywords = user_input.split()

    for data_item in stored_data:
        subtheme_keywords = extract_keywords_from_subtheme(data_item['subtheme'])

        # Check if any keyword in user input matches subtheme keywords
        for user_keyword in user_keywords:
            for subtheme_keyword in subtheme_keywords:
                similarity = SequenceMatcher(None, user_keyword, subtheme_keyword).ratio()

                # Exact match or close match
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = data_item['response']

    # Return response if there's a strong enough match, otherwise None
    return best_match if highest_similarity > 0.3 else None  # Adjust threshold as needed

# Query Neo4j for the response based on the keyword
def get_response_for_subtheme(user_input):
    driver = connect_to_neo4j()
    stored_data = []
    query = """
    MATCH (s:Subtheme)-[:HAS_RESPONSE]->(r:Response)
    RETURN s.name AS subtheme, r.text AS response
    """
    
    with driver.session() as session:
        results = session.run(query)
        for record in results:
            stored_data.append({'subtheme': record['subtheme'], 'response': record['response']})
    
    response = match_user_input(user_input, stored_data)
    return response if response else "Sorry, I couldn't find a relevant response in the database."

# Create date-specific log filenames
log_dir = 'chat_logs'
os.makedirs(log_dir, exist_ok=True)  # Ensure the chat_logs directory exists

current_date = datetime.now().strftime("%Y-%m-%d")
log_filename_csv = os.path.join(log_dir, f"chat_logs_{current_date}.csv")
log_filename_txt = os.path.join(log_dir, f"chat_logs_{current_date}.txt")

# Set up logs for CSV file
if not os.path.exists(log_filename_csv):
    with open(log_filename_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Response', 'Response Time'])  # Write header row if file is new

# Set up logs for TXT file
logging.basicConfig(filename=log_filename_txt, level=logging.INFO, format='%(asctime)s - %(message)s')

# Assistant Class
class Assistant:
    def __init__(self, context):
        self.context = context
        self.chat_history = []

    # Process user input and generate a response
    def process_chat(self, question):
        start_time = datetime.now()  # Timestamp when question is received
        neo4j_response = get_response_for_subtheme(question)
        response = neo4j_response
        logging.info("Response source: Neo4j")

        end_time = datetime.now()  # Timestamp after response is generated
        response_time = (end_time - start_time).total_seconds()

        self.log_to_csv(question, response, response_time)
        self.log_chat_history(question, response)
        self.chat_history.append({'question': question, 'response': response})
        
        print(f"Response Time: {response_time} seconds")  # Print response time
        return response

    # Log to CSV file
    def log_to_csv(self, question, answer, response_time):
        with open(log_filename_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([question, answer, response_time])

    # Log chat entry to TXT file
    def log_chat_history(self, question, answer):
        logging.info(f"User: {question}")
        logging.info(f"Assistant: {answer}")

    # Reset chat history
    def reset_chat_history(self):
        self.chat_history = []

# Main execution
if __name__ == '__main__':
    assistant = Assistant('map navigation')
    print("Hello! Welcome to the Atlas Map Navigation Assistant!")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Thank you for using the Atlas Map Navigation Assistant. Goodbye!")
            break

        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Could you rephrase your question?")
