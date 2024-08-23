from flask import Flask, request
from flask_cors import CORS
from langchain_sql import Langchain_SQL
from qna_with_neo4j import QnA_with_Neo4j

app = Flask(__name__)
CORS(app, origins=["*"])

langchain_sql = Langchain_SQL()
qna_with_neo4j = QnA_with_Neo4j()

@app.route('/', methods=['POST'])
def test():
    return 'testing works'

@app.route('/test', methods=['POST'])
def test2():
    response = langchain_sql.test()

    return response

@app.route('/get_table_from_query', methods=['POST'])
def get_table_from_query():
    input = request.get_json().get('input')
    response = langchain_sql.get_table_from_query(input)

    return response

@app.route('/get_relevent_tables_from_category', methods=['POST'])
def get_relevent_tables_from_category():
    input = request.get_json().get('input')
    response = langchain_sql.get_relevent_tables_from_category(input)

    return response

@app.route('/get_response_from_sql_chain', methods=['POST'])
def get_response_from_sql_chain():
    input = request.get_json().get('input')
    response = langchain_sql.get_response_from_sql_chain(input)

    return response

@app.route('/high_cardinality_columns', methods=['POST'])
def high_cardinality_columns():
    input = request.get_json().get('input')
    response = langchain_sql.high_cardinality_columns(input)

    return response

@app.route('/qna_with_neo4j', methods=['POST'])
def initialize_data():
    response =qna_with_neo4j.initialize_data()

    return response


@app.route('/qna_with_neo4j/get_schema', methods=['POST'])
def get_schema():
    qna_with_neo4j.get_schema()

    return "OK"

@app.route('/qna_with_neo4j/chain_response', methods=['POST'])
def get_chain_response():
    input = request.get_json().get('input')
    response = qna_with_neo4j.get_response(input)

    return response

@app.route('/qna_with_neo4j/detect_entities_chain', methods=['POST'])
def detect_entities():
    input = request.get_json().get('input')
    response = qna_with_neo4j.detect_entities(input)

    return response

@app.route('/qna_with_neo4j/generate_cypher_query', methods=['POST'])
def generate_cypher_query():
    input = request.get_json().get('input')
    response = qna_with_neo4j.generate_cypher_query(input)

    return response

@app.route('/qna_with_neo4j/get_response_from_cypher_query', methods=['POST'])
def get_response_from_cypher_query():
    input = request.get_json().get('input')
    response = qna_with_neo4j.get_response_from_generated_cypher_query(input)

    return response

@app.route('/knowledge_graph', methods=['POST'])
def get_knowledge_graph():
    response = qna_with_neo4j.knowledge_graph()

    return response

@app.route('/qna_with_neo4j/test', methods=['POST'])
def test_embed():
    response = qna_with_neo4j.embedding_story_to_graph()

    return response

@app.route('/qna_with_neo4j/retrieve_docs_from_query', methods=['POST'])
def retrieve_docs_from_query():
    input = request.get_json().get('input')
    response = qna_with_neo4j.retrieve_docs_from_query(input)

    return response

if __name__ == '__main__':
    app.run(port=8080)
