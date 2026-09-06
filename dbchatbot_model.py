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

# ---------------------------------------------------------

@tool
def query_students(
    order_by: str = "id",
    order_direction: str = "ASC",
    branch: str = None,
    min_age: int = None,
    max_age: int = None,
    limit: int = None
):
    """
    Get students from the students table with sorting,
    filtering, and limiting options. Use this instead of
    get_students whenever the user asks to sort, filter,
    or limit the results in any way.

    Args:
        order_by: Column to sort by. One of:
            "id", "name", "age", "branch". Defaults to "id".
        order_direction: "ASC" or "DESC". Defaults to "ASC".
        branch: Optional. Filter to only this branch
            (e.g. "IT", "CSE"). Leave unset for all branches.
        min_age: Optional. Only students at or above this age.
        max_age: Optional. Only students at or below this age.
        limit: Optional. Max number of rows to return
            (e.g. "top 5 oldest students" -> limit=5).

    Examples:
        - "show students ordered by age" -> order_by="age"
        - "students in IT branch" -> branch="IT"
        - "top 5 oldest students" -> order_by="age", order_direction="DESC", limit=5
        - "students older than 20" -> min_age=21
        - "CSE students sorted by name" -> branch="CSE", order_by="name"
    """

    allowed_columns = {"id", "name", "age", "branch"}
    allowed_directions = {"ASC", "DESC"}

    if order_by not in allowed_columns:
        order_by = "id"

    order_direction = order_direction.upper()
    if order_direction not in allowed_directions:
        order_direction = "ASC"

    # Build WHERE clause dynamically using placeholders for
    # actual values (safe from SQL injection), while column
    # names / keywords are whitelisted separately above.
    conditions = []
    params = []

    if branch is not None:
        conditions.append("branch = %s")
        params.append(branch)

    if min_age is not None:
        conditions.append("age >= %s")
        params.append(min_age)

    if max_age is not None:
        conditions.append("age <= %s")
        params.append(max_age)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT %s"
        params.append(limit)

    query = f"""
        SELECT id, name, age, branch
        FROM students
        {where_clause}
        ORDER BY {order_by} {order_direction}
        {limit_clause};
    """

    with conn.cursor() as cur:
        cur.execute(query, params)
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
# =========================================================
# TOOL LIST
# =========================================================

tools = [
    create_students_table,
    insert_student,
    get_students,
    update_student,
    delete_student,
    query_students
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
# NOTE: Annotated[list, add_messages] is the fix.
# Without the add_messages reducer, LangGraph OVERWRITES
# the messages list on every node return instead of
# appending to it -- so by the time workflow.invoke()
# finishes, result["messages"] only contains the LAST
# message, and your ToolMessage from get_students is lost
# before FastAPI ever sees it.

class DBState(TypedDict):

    messages: Annotated[list, add_messages]


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