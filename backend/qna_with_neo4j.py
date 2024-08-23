import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person or movies appearing in the text",
    )

class QnA_with_Neo4j:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        if not api_key:
            raise ValueError(
                "API_KEY is missing from the environment variables.")
        
        self.graph = Neo4jGraph()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)


    def initialize_data(self):
        try:
            # movies_query = """
            # LOAD CSV WITH HEADERS FROM 
            # 'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
            # AS row
            # MERGE (m:Movie {id:row.movieId})
            # SET m.released = date(row.released),
            #     m.title = row.title,
            #     m.imdbRating = toFloat(row.imdbRating)
            # FOREACH (director in split(row.director, '|') | 
            #     MERGE (p:Person {name:trim(director)})
            #     MERGE (p)-[:DIRECTED]->(m))
            # FOREACH (actor in split(row.actors, '|') | 
            #     MERGE (p:Person {name:trim(actor)})
            #     MERGE (p)-[:ACTED_IN]->(m))
            # FOREACH (genre in split(row.genres, '|') | 
            #     MERGE (g:Genre {name:trim(genre)})
            #     MERGE (m)-[:IN_GENRE]->(g))
            # """

            # print(self.graph.query(movies_query))

            # print(self.graph.query("DROP DATABASE "))

            with open("short_history_1.txt") as f:
                story = f.read()

            text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ".",
                    ",",
                    "\u200b",  # Zero-width space
                    "\uff0c",  # Fullwidth comma
                    "\u3001",  # Ideographic comma
                    "\uff0e",  # Fullwidth full stop
                    "\u3002",  # Ideographic full stop
                    "",
                ],
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )

            self.texts = text_splitter.create_documents([story])
            self.db = Neo4jVector.from_documents(
            self.texts, self.embeddings, url=self.uri, username=self.user, password=self.password
            )

            return "Text chunked embedded and saved to the database"

        except Exception as e:
            return str(e)

    def get_schema(self):
        try:
            self.graph.refresh_schema()
            print(self.graph.schema)

            # Schema printed in the terminal:
            # Node properties:
            # Movie {imdbRating: FLOAT, id: STRING, released: DATE, title: STRING}
            # Person {name: STRING}
            # Genre {name: STRING}
            # Relationship properties:

            # The relationships:
            # (:Movie)-[:IN_GENRE]->(:Genre)
            # (:Person)-[:DIRECTED]->(:Movie)
            # (:Person)-[:ACTED_IN]->(:Movie)

        except Exception as e:
            print(e)

    def get_response(self, input: str):
        try:
            chain = GraphCypherQAChain.from_llm(graph=self.graph, llm=self.llm, verbose=True)
            response = chain.invoke({"query": input})
            print(response)
            # Response: {'query': 'What was the cast of the Casino?', 'result': 'The cast of Casino included James Woods, Joe Pesci, Robert De Niro, and Sharon Stone.'}
            return response

        except Exception as e:
            print(e)
            return str(e)
        
    def detect_entities(self, input: str):
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are extracting person and movies from the text.",
                    ),
                    (
                        "human",
                        "Use the given format to extract information from the following "
                        "input: {question}",
                    ),
                ]
            )


            entity_chain = prompt | self.llm.with_structured_output(Entities)
            entities = entity_chain.invoke({"question": input})

            match_query = """MATCH (p:Person|Movie)
            WHERE p.name CONTAINS $value OR p.title CONTAINS $value
            RETURN coalesce(p.name, p.title) AS result, labels(p)[0] AS type
            LIMIT 1
            """
            
            print(entities)
            return self.map_to_database(match_query, entities)

        except Exception as e:
            return str(e)
        
    def generate_cypher_query(self, input: str):
        try:
            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         (
            #             "system",
            #             "You are extracting person and movies from the text.",
            #         ),
            #         (
            #             "human",
            #             "Use the given format to extract information from the following "
            #             "input: {question}",
            #         ),
            #     ]
            # )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are extracting question, user, tags, or comments from the text.",
                    ),
                    (
                        "human",
                        "Use the given format to extract information from the following "
                        "input: {question}",
                    ),
                ]
            )


            entity_chain = prompt | self.llm.with_structured_output(Entities)

            # match_query = """MATCH (p:Person|Movie)
            # WHERE p.name CONTAINS $value OR p.title CONTAINS $value
            # RETURN coalesce(p.name, p.title) AS result, labels(p)[0] AS type
            # LIMIT 1
            # """

            match_query = """MATCH (p:Question|User|Tag)
            WHERE p.name CONTAINS $value OR p.title CONTAINS $value
            RETURN coalesce(p.name, p.title) AS result, labels(p)[0] AS type
            LIMIT 1
            """

            # Generate Cypher statement based on natural language input
            cypher_template = """Based on the Neo4j graph schema below, write a correct and valid Cypher query that would answer the user's question:
            {schema}
            Entities in the question map to the following database values:
            {entities_list}
            Question: {question}
            Cypher query:"""

            cypher_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given an input question, convert it to a correct Cypher query. No pre-amble.",
                    ),
                    ("human", cypher_template),
                ]
            )

            cypher_response = (
                RunnablePassthrough.assign(names=entity_chain)
                | RunnablePassthrough.assign(
                    entities_list=lambda x: self.map_to_database(match_query, x["names"]),
                    schema=lambda _: self.graph.get_schema,
                )
                | cypher_prompt
                | self.llm.bind(stop=["\nCypherResult:"])
                | StrOutputParser()
            )

            cypher = cypher_response.invoke({"question": input})
            return cypher
        # Example output
        # MATCH (m:Movie)
        # WHERE m.title IN ['Jumanji', 'Toy Story', 'Casino']
        # RETURN m.title, m.imdbRating, m.released

        # MATCH (p:Person)
        # WHERE p.name = 'James Woods'
        # RETURN p.name

        except Exception as e:
            return str(e)
    
    def get_response_from_generated_cypher_query(self, input: str):
        try:
            # prompt = ChatPromptTemplate.from_messages(
            #         [
            #             (
            #                 "system",
            #                 "You are extracting person and movies from the text.",
            #             ),
            #             (
            #                 "human",
            #                 "Use the given format to extract information from the following "
            #                 "input: {question}",
            #             ),
            #         ]
            #     )

            prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are extracting question and user from the text.",
                        ),
                        (
                            "human",
                            "Use the given format to extract information from the following "
                            "input: {question}",
                        ),
                    ]
                )


            entity_chain = prompt | self.llm.with_structured_output(Entities)

            match_query = """MATCH (p:Question|User)
            WHERE p.name CONTAINS $value OR p.title CONTAINS $value
            RETURN coalesce(p.name, p.title) AS result, labels(p)[0] AS type
            LIMIT 1
            """

            # Generate Cypher statement based on natural language input
            cypher_template = """Based on the Neo4j graph schema below, write a Cypher query that would answer the user's question:
            {schema}
            Entities in the question map to the following database values:
            {entities_list}
            Question: {question}
            Cypher query:"""

            cypher_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given an input question, convert it to a Cypher query. No pre-amble.",
                    ),
                    ("human", cypher_template),
                ]
            )

            cypher_response = (
                RunnablePassthrough.assign(names=entity_chain)
                | RunnablePassthrough.assign(
                    entities_list=lambda x: self.map_to_database(match_query, x["names"]),
                    schema=lambda _: self.graph.get_schema,
                )
                | cypher_prompt
                | self.llm.bind(stop=["\nCypherResult:"])
                | StrOutputParser()
            )

            # print(cypher_response)

            # Cypher validation tool for relationship directions
            corrector_schema = [
                Schema(el["start"], el["type"], el["end"])
                for el in self.graph.structured_schema.get("relationships")
            ]
            cypher_validation = CypherQueryCorrector(corrector_schema)

            # Generate natural language response based on database results
            response_template = """Based on the the question, Cypher query, and Cypher response, write a natural language response:
            Question: {question}
            Cypher query: {query}
            Cypher Response: {response}"""

            response_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given an input question and Cypher response, convert it to a natural"
                        " language answer. No pre-amble.",
                    ),
                    ("human", response_template),
                ]
            )

            chain = (
                RunnablePassthrough.assign(query=cypher_response)
                | RunnablePassthrough.assign(
                    response=lambda x: self.graph.query(cypher_validation(x["query"])),
                )
                | response_prompt
                | self.llm
                | StrOutputParser()
            )

            response = chain.invoke({"question": input})
            return response
        
        except Exception as e:
            return str(e)

    def map_to_database(self, match_query: str, entities: Entities) -> Optional[str]:
        result = ""
        for entity in entities.names:
            print("entity:", entity)
            response = self.graph.query(match_query, {"value": entity})
            print("response:", response)
            try:
                result += f"{entity} maps to {response[0]['result']} {response[0]['type']} in database\n"
                print("result:", result)
            except IndexError:
                print("IndexError, skipping")
                pass
        return result
    
    # def sql_to_neo4j(self, input: str):
    #     try:
    #         prompt_template = """
    #         You are an expert in relational databases and graph databases. You are given an SQL schema, you need to convert
    #         it to a Neo
    #         """

    def knowledge_graph(self):

        # docs = WikipediaLoader(query=input, load_max_docs=1).load()

        # text = """
        # Marie Curie, born in 1867, was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.
        # She was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.
        # Her husband, Pierre Curie, was a co-winner of her first Nobel Prize, making them the first-ever married couple to win the Nobel Prize and launching the Curie family legacy of five Nobel Prizes.
        # She was, in 1906, the first woman to become a professor at the University of Paris.
        # """
        # documents = [Document(page_content=text)]
        graph_documents = self.llm_transformer.convert_to_graph_documents(self.texts)
        # print(f"Nodes:{graph_documents[0].nodes}")
        # print(f"Relationships:{graph_documents[0].relationships}")

        # # Output:
        # # Nodes:[Node(id='Marie Curie', type='Person'), Node(id='Pierre Curie', type='Person'), Node(id='University Of Paris',
        # # type='University')]

        # # Relationships:[Relationship(source=Node(id='Marie Curie', type='Person'), target=Node(id='Pierre Curie', type='Person'),
        # # type='SPOUSE'), Relationship(source=Node(id='Marie Curie', type='Person'), target=Node(id='University Of Paris',
        # # type='University'), type='PROFESSOR')]

        # llm_transformer_filtered = LLMGraphTransformer(
        #     llm=self.llm,
        #     allowed_nodes=["Person", "Country", "Organization", "Document"],
        #     allowed_relationships=["NATIONALITY", "WROTE_IN"],
        # )
        # graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(
        #     docs
        # )

        self.graph.add_graph_documents(graph_documents)

        return f"""
        Nodes:{graph_documents[0].nodes}
        Relationships:{graph_documents[0].relationships}
        """
        # # Output:
        # # Nodes:[Node(id='Marie Curie', type='Person'), Node(id='Poland', type='Country'), Node(id='France', type='Country'),
        # # Node(id='Pierre Curie', type='Person'), Node(id='University Of Paris', type='Organization')]

        # # Relationships:[Relationship(source=Node(id='Marie Curie', type='Person'), target=Node(id='Poland', type='Country'),
        # # type='NATIONALITY'), Relationship(source=Node(id='Marie Curie', type='Person'), target=Node(id='France',
        # # type='Country'), type='NATIONALITY'), Relationship(source=Node(id='Marie Curie', type='Person'),
        # # target=Node(id='University Of Paris', type='Organization'), type='WORKED_AT'), Relationship(source=Node(id='Marie
        # # Curie', type='Person'), target=Node(id='Pierre Curie', type='Person'), type='SPOUSE')]

        # llm_transformer_props = LLMGraphTransformer(
        #     llm=self.llm,
        #     allowed_nodes=["Person", "Country", "Organization"],
        #     allowed_relationships=["NATIONALITY", "LOCATED_IN", "WORKED_AT", "SPOUSE"],
        #     node_properties=["born_year"],
        # )
        # graph_documents_props = llm_transformer_props.convert_to_graph_documents(documents)

        # # Output:
        # # Nodes:[Node(id='Marie Curie', type='Person', properties={'born_year': '1867'}), Node(id='Polish', type='Country'),
        # # Node(id='French', type='Country'), Node(id='Physicist', type='Organization'), Node(id='Chemist', type='Organization'),
        # # Node(id='Pierre Curie', type='Person'), Node(id='University Of Paris', type='Organization')]

        # # Relationships:[Relationship(source=Node(id='Marie Curie', type='Person'), target=Node(id='Polish', type='Country'),
        # # type='NATIONALITY'), Relationship(source=Node(id='Marie Curie', type='Person'), target=Node(id='French',
        # # type='Country'), type='NATIONALITY'), Relationship(source=Node(id='Marie Curie', type='Person'),
        # # target=Node(id='Physicist', type='Organization'), type='WORKED_AT'), Relationship(source=Node(id='Marie Curie',
        # # type='Person'), target=Node(id='Chemist', type='Organization'), type='WORKED_AT'), Relationship(source=Node(id='Marie
        # # Curie', type='Person'), target=Node(id='Pierre Curie', type='Person'), type='SPOUSE')]

        # self.graph.add_graph_documents(graph_documents_props)

        # return f"""
        # Nodes:{graph_documents_props[0].nodes}

        # Relationships:{graph_documents_props[0].relationships}
        # """

    def embedding_story_to_graph(self):
        try:
            llm_transformer_props = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=["Character"],
                allowed_relationships=["SPOUSE", "FRIENDS_WITH", "ENEMIES_WITH"],
            )
            graph_documents_props = llm_transformer_props.convert_to_graph_documents(self.texts)
            print(len(graph_documents_props))

            for props in graph_documents_props:
                print(f"Nodes:{props.nodes}")
                print(f"Relationships:{props.relationships}")
                print()

            self.graph.add_graph_documents(graph_documents_props)
            # for text in texts:
            #     print(text)
            #     print()

            return "OK"

        except Exception as e:
            return str(e)
        
    def retrieve_docs_from_query(self, input: str):
        try:

            texts_with_score = self.db.similarity_search_with_score(input, k=3)

            for text, score in texts_with_score:
                print("-" * 80)
                print("Score: ", score)
                print(text.page_content)
                print("-" * 80)

            prompt = """

            """

            return "OK"

        except Exception as e:
            return str(e)

            