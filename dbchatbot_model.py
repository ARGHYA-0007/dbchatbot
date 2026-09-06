import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, BaseMessage
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.prebuilt import ToolNode

from langchain_core.tools import tool
import requests
import sqlite3
import psycopg 
from langchain_ollama import OllamaEmbeddings
import json
import time
from langchain_ollama import ChatOllama

import numpy as np
load_dotenv()

llm = ChatOllama(model = 'qwen2.5:7b')


conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)
@tool
def create_students_table():
    """
    Create the students table.

    Columns:
    id, name, age, branch
    """

    with conn.cursor() as cur:

        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                age INTEGER,
                branch VARCHAR(100)
            );
        """)

    conn.commit()

    return "Students table created successfully."


# ---------------------------------------------------------

@tool
def insert_student(name: str, age: int, branch: str):
    """
    Insert a student into the students table.
    """

    with conn.cursor() as cur:

        cur.execute(
            """
            INSERT INTO students
            (name, age, branch)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (name, age, branch)
        )

        student_id = cur.fetchone()[0]

    conn.commit()

    return f"Student added successfully with ID {student_id}."


# ---------------------------------------------------------

@tool
def get_students():
    """
    Get all students from the students table.
    """

    with conn.cursor() as cur:

        cur.execute("""
            SELECT id, name, age, branch
            FROM students
            ORDER BY id;
        """)

        rows = cur.fetchall()

    students = []

    for row in rows:

        students.append({
            "id": row[0],
            "name": row[1],
            "age": row[2],
            "branch": row[3]
        })

    return json.dumps(students)


# ---------------------------------------------------------

@tool
def update_student(
    student_id: int,
    branch: str
):
    """
    Update the branch of a student.
    """

    with conn.cursor() as cur:

        cur.execute(
            """
            UPDATE students
            SET branch = %s
            WHERE id = %s
            RETURNING id;
            """,
            (branch, student_id)
        )

        result = cur.fetchone()

    conn.commit()

    if result is None:
        return "Student not found."

    return f"Student {student_id} updated successfully."


# ---------------------------------------------------------

@tool
def delete_student(student_id: int):
    """
    Delete a student by ID.
    """

    with conn.cursor() as cur:

        cur.execute(
            """
            DELETE FROM students
            WHERE id = %s
            RETURNING id;
            """,
            (student_id,)
        )

        result = cur.fetchone()

    conn.commit()

    if result is None:
        return "Student not found."

    return f"Student {student_id} deleted successfully."


# =========================================================
# TOOL LIST
# =========================================================

tools = [
    create_students_table,
    insert_student,
    get_students,
    update_student,
    delete_student
]


# =========================================================
# LLM
# =========================================================

# llm = ChatOllama(
#     model="qwen2.5:1.5b"
# )

llm_with_tools = llm.bind_tools(tools)


# =========================================================
# STATE
# =========================================================

class DBState(TypedDict):

    messages: list


# =========================================================
# CHAT NODE
# =========================================================

def chatbot(state: DBState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


# =========================================================
# CREATE GRAPH
# =========================================================

graph = StateGraph(DBState)


graph.add_node(
    "chatbot",
    chatbot
)


graph.add_node(
    "tools",
    ToolNode(tools)
)


graph.add_edge(
    START,
    "chatbot"
)


# =========================================================
# CONDITIONAL ROUTING
# =========================================================

def should_continue(state: DBState):

    last_message = state["messages"][-1]

    # If LLM requested a tool
    if last_message.tool_calls:

        return "tools"

    # Otherwise finish
    return END


graph.add_conditional_edges(
    "chatbot",
    should_continue
)


graph.add_edge(
    "tools",
    "chatbot"
)


workflow = graph.compile()


# # =========================================================
# # CHAT LOOP
# # =========================================================

# print("=" * 60)
# print("        PostgreSQL AI DATABASE AGENT")
# print("=" * 60)

# print("\nType 'exit' to stop.\n")


# while True:

#     user_input = input("You: ")

#     if user_input.lower() == "exit":
#         break

#     result = workflow.invoke(
#         {
#             "messages": [
#                 HumanMessage(
#                     content=user_input
#                 )
#             ]
#         }
#     )

#     final_message = result["messages"][-1]

#     print("\nAI:", final_message.content)
#     print()


# =========================================================
# CLOSE DATABASE
# =========================================================

# conn.close()