import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from typing import TypedDict
from operator import itemgetter
from langchain.chains import create_sql_query_chain
from langchain_core.runnables import RunnablePassthrough, RunnableMap
import ast
import re
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

# class Table_List(TypedDict):
#     name: str

class Langchain_SQL:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "API_KEY is missing from the environment variables.")

        self.llm = ChatOpenAI(
            temperature=0.0, model="gpt-4o-mini", api_key=api_key
        )
        self.db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        self.embeddings = OpenAIEmbeddings(api_key=api_key)

    def test(self):
        print(self.db.dialect)
        print(self.db.get_usable_table_names())
        print(self.db.run("SELECT * FROM Artist LIMIT 10;"))

        return "OK"
    
    def get_table_from_query(self, input: str):
        print()
        print("This is the input: ", input)
        print()
        table_names = "\n".join(self.db.get_usable_table_names())

        system = f"""Return the names of ALL the SQL tables that are ONLY relevant to the user question. \
        The tables are:

        {table_names}

        Remember to include ONLY RELEVANT TABLES."""

        system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
        The tables are:

        {table_names}

        Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{input}"),
            ]
        )
        llm_with_tools = self.llm.bind_tools([Table])
        output_parser = PydanticToolsParser(tools=[Table])

        table_chain = prompt | llm_with_tools | output_parser
        response = str(table_chain.invoke({"input": {input}})) # Prints tables that are relevant to the user's input, but there could be a possibility of tables that were not included. 

        return response
    
    def get_relevent_tables_from_category(self, input: str):
        print()
        print("This is the input: ", input)
        print()

        llm_with_tools = self.llm.bind_tools([Table])
        output_parser = PydanticToolsParser(tools=[Table])

        system = """Return the names of any SQL tables that are relevant to the user question.
        The tables are:

        Music
        Business

        Remember to choose one of the categories that is most relevent to the user's input.
        """

        # I think this can be useful if you have a long list of categories that certain tables fit into. I had to specify in the prompt that the model needs to choose one of the categories.
        # Without it, the model will always list all of the tables

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{input}"),
            ]
        )

        category_chain = prompt | llm_with_tools | output_parser

        table_chain = category_chain | self.get_tables 
        response = str(table_chain.invoke({"input": {input}})) # Prints a list of table names from the category that is relevent to the user's input

        return response


    def get_tables(self, categories: List[Table]) -> List[str]:
        tables = []
        for category in categories:
            if category.name == "Music":
                tables.extend(
                    [
                        "Album",
                        "Artist",
                        "Genre",
                        "MediaType",
                        "Playlist",
                        "PlaylistTrack",
                        "Track",
                    ]
                )
            elif category.name == "Business":
                tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
        return tables
    
    def get_response_from_sql_chain(self, input: str):
        print()
        print("This is the input: ", input)
        print()

        llm_with_tools = self.llm.bind_tools([Table])
        output_parser = PydanticToolsParser(tools=[Table])

        system = """Return the names of any SQL tables that are relevant to the user question.
        The tables are:

        Music
        Business

        Remember to choose one of the categories that is most relevent to the user's input.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{input}"),
            ]
        )

        category_chain = prompt | llm_with_tools | output_parser

        table_chain = category_chain | self.get_tables

        query_chain = create_sql_query_chain(self.llm, self.db)

        table_chain = {"input": itemgetter("question")} | table_chain

        # print(type(table_chain.invoke({"question": {input}})))

        full_chain = RunnablePassthrough.assign(table_names_to_use=table_chain) | query_chain

        query = full_chain.invoke(
            {"question": input}
        )
        print(query) # Prints the SQL query that is generated from the user's input

        print(self.db.run(query)) # Currently, prints an operational error.

        # sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) near "SQLQuery": syntax error
        # [SQL: SQLQuery: 
        # ```sql
        # SELECT DISTINCT "Genre"."Name" 
        # FROM "Track" 
        # JOIN "Album" ON "Track"."AlbumId" = "Album"."AlbumId" 
        # JOIN "Artist" ON "Album"."ArtistId" = "Artist"."ArtistId" 
        # JOIN "Genre" ON "Track"."GenreId" = "Genre"."GenreId" 
        # WHERE "Artist"."Name" = 'Alanis Morissette' 
        # LIMIT 5;
        # ```]
        # (Background on this error at: https://sqlalche.me/e/20/e3q8)

        return query
    
    def query_as_list(self,db, query):
        res = db.run(query)
        res = [el for sub in ast.literal_eval(res) for el in sub if el]
        res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
        return res
    
    def high_cardinality_columns(self, input: str):
        user_input = input
        proper_nouns = self.query_as_list(self.db, "SELECT Name FROM Artist")
        proper_nouns += self.query_as_list(self.db, "SELECT Title FROM Album")
        proper_nouns += self.query_as_list(self.db, "SELECT Name FROM Genre")
        print(len(proper_nouns)) # Result was 647 proper nouns
        print(proper_nouns[:5]) # ['AC/DC', 'Accept', 'Aerosmith', 'Alanis Morissette', 'Alice In Chains']
        print()
        print("This is the input: ", input)
        print()

        vector_db = FAISS.from_texts(proper_nouns, self.embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 15})

        system = """You are a SQLite expert. Given an input question, create a syntactically
        correct SQLite query to run. Unless otherwise specificed, do not return more than
        {top_k} rows.

        Only return the SQL query with no markup or explanation.

        Here is the relevant table info: {table_info}

        Here is a non-exhaustive list of possible feature values. If filtering on a feature
        value make sure to check its spelling against this list first:

        {proper_nouns}
        """

        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

        query_chain = create_sql_query_chain(self.llm, self.db, prompt=prompt)
        retriever_chain = (
            itemgetter("question")
            | retriever
            | (lambda docs: "\n".join(doc.page_content for doc in docs))
        )
        chain = RunnablePassthrough.assign(proper_nouns=retriever_chain) | query_chain

        print("First using query_chain without retriever")
        query = query_chain.invoke(
            {"question": user_input, "proper_nouns": ""}
        )
        print("this is the query:")
        print(query)

        print("Running the query into the database we get:")
        query_res =self.db.run(query)

        print("Now using query_chain with retriever")
        query1 = chain.invoke({"question": user_input})
        print("this is the query:")
        print(query1)
        print("Running the query into the database we get:")
        query_res1 = self.db.run(query1)

        return f"""

        This is the input: {input}

        First using query_chain without retriever:
        {query}

        Running the query into the database we get:
        {query_res}

        Now using query_chain with retriever:
        {query1}

        Running the query into the database we get:
        {query_res1}
        """