"""
CodeMentor AI - ChromaDB Collection Definitions
=================================================
Defines all 15+ knowledge base collections used by the RAG system.

Design Rationale:
- Domain-specific collections enable targeted retrieval.
  Instead of searching "all knowledge", the agent can search
  "python_docs" when answering Python questions — higher precision.
- Collection metadata captures topic, language, and content type
  for filtering at query time.
- HNSW (Hierarchical Navigable Small World) is ChromaDB's default
  ANN index — fast approximate nearest neighbor search.
"""

from dataclasses import dataclass, field
from enum import Enum


class ContentType(str, Enum):
    """Types of content stored in collections."""
    DOCUMENTATION = "documentation"
    DSA_NOTES = "dsa_notes"
    ALGORITHMS = "algorithms"
    INTERVIEW = "interview"
    BEST_PRACTICES = "best_practices"
    ERRORS = "errors"
    SYSTEM_DESIGN = "system_design"
    OS = "operating_systems"
    DBMS = "dbms"
    NETWORKS = "computer_networks"
    GENERAL = "general"


class ProgrammingLanguage(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVA = "java"
    CPP = "cpp"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    GENERAL = "general"


@dataclass
class CollectionConfig:
    """
    Configuration for a single ChromaDB collection.

    Attributes:
        name:        ChromaDB collection name (unique identifier).
        description: Human-readable description for Swagger/admin UI.
        topic:       Broad topic category.
        language:    Associated programming language (if applicable).
        content_type: Type of content in this collection.
    """
    name: str
    description: str
    topic: str
    language: ProgrammingLanguage = ProgrammingLanguage.GENERAL
    content_type: ContentType = ContentType.DOCUMENTATION
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = {
            "topic": self.topic,
            "language": self.language.value,
            "content_type": self.content_type.value,
            "description": self.description,
        }


# ==============================================================
# All RAG Collections
# ==============================================================
# These are the 15 ChromaDB collections for CodeMentor AI.
# Each maps to a domain of programming knowledge.
# ==============================================================

ALL_COLLECTIONS: list[CollectionConfig] = [

    # ------ Programming Language Documentation ------
    CollectionConfig(
        name="python_docs",
        description="Official Python 3.x documentation — stdlib, built-ins, typing, async",
        topic="python_documentation",
        language=ProgrammingLanguage.PYTHON,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="java_docs",
        description="Java SE and JDK documentation — collections, streams, generics, OOP",
        topic="java_documentation",
        language=ProgrammingLanguage.JAVA,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="cpp_docs",
        description="C++ STL, pointers, memory management, templates, OOP documentation",
        topic="cpp_documentation",
        language=ProgrammingLanguage.CPP,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="javascript_docs",
        description="JavaScript ES6+, DOM, async/await, promises, Node.js documentation",
        topic="javascript_documentation",
        language=ProgrammingLanguage.JAVASCRIPT,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="sql_docs",
        description="SQL queries, joins, indexes, transactions, optimization documentation",
        topic="sql_documentation",
        language=ProgrammingLanguage.SQL,
        content_type=ContentType.DOCUMENTATION,
    ),

    # ------ Framework Documentation ------
    CollectionConfig(
        name="fastapi_docs",
        description="FastAPI framework — routing, dependencies, middleware, OpenAPI, async",
        topic="fastapi_documentation",
        language=ProgrammingLanguage.PYTHON,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="sqlalchemy_docs",
        description="SQLAlchemy ORM — models, sessions, queries, relationships, migrations",
        topic="sqlalchemy_documentation",
        language=ProgrammingLanguage.PYTHON,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="docker_docs",
        description="Docker — containers, images, Dockerfile, compose, networking, volumes",
        topic="docker_documentation",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.DOCUMENTATION,
    ),
    CollectionConfig(
        name="git_docs",
        description="Git version control — branching, merging, rebase, workflows, hooks",
        topic="git_documentation",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.DOCUMENTATION,
    ),

    # ------ CS Fundamentals ------
    CollectionConfig(
        name="dsa_notes",
        description="Data Structures and Algorithms — arrays, trees, graphs, sorting, DP",
        topic="data_structures_algorithms",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.DSA_NOTES,
    ),
    CollectionConfig(
        name="algorithms",
        description="Algorithm design — divide & conquer, greedy, backtracking, complexity",
        topic="algorithms",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.ALGORITHMS,
    ),
    CollectionConfig(
        name="system_design",
        description="System design — scalability, CAP theorem, microservices, load balancing",
        topic="system_design",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.SYSTEM_DESIGN,
    ),
    CollectionConfig(
        name="os_notes",
        description="Operating Systems — processes, threads, scheduling, memory, deadlocks",
        topic="operating_systems",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.OS,
    ),
    CollectionConfig(
        name="dbms_notes",
        description="DBMS — normalization, transactions, ACID, indexing, query optimization",
        topic="database_management",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.DBMS,
    ),
    CollectionConfig(
        name="networks_notes",
        description="Computer Networks — TCP/IP, HTTP, DNS, OSI model, sockets",
        topic="computer_networks",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.NETWORKS,
    ),

    # ------ Career & Interview ------
    CollectionConfig(
        name="interview_qna",
        description="Programming interview questions, answers, and common patterns",
        topic="interview_preparation",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.INTERVIEW,
    ),
    CollectionConfig(
        name="best_practices",
        description="Programming best practices — SOLID, DRY, clean code, design patterns",
        topic="best_practices",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.BEST_PRACTICES,
    ),
    CollectionConfig(
        name="common_errors",
        description="Common programming errors, bug patterns, debugging strategies",
        topic="debugging_errors",
        language=ProgrammingLanguage.GENERAL,
        content_type=ContentType.ERRORS,
    ),
]

# Lookup map: collection name → config
COLLECTION_MAP: dict[str, CollectionConfig] = {
    col.name: col for col in ALL_COLLECTIONS
}

# All valid collection names (for API validation)
VALID_COLLECTION_NAMES: list[str] = [col.name for col in ALL_COLLECTIONS]

# Topic → collection name mapping (for agent routing decisions)
TOPIC_TO_COLLECTION: dict[str, str] = {
    "python": "python_docs",
    "java": "java_docs",
    "c++": "cpp_docs",
    "cpp": "cpp_docs",
    "javascript": "javascript_docs",
    "js": "javascript_docs",
    "sql": "sql_docs",
    "fastapi": "fastapi_docs",
    "sqlalchemy": "sqlalchemy_docs",
    "docker": "docker_docs",
    "git": "git_docs",
    "dsa": "dsa_notes",
    "data structures": "dsa_notes",
    "algorithms": "algorithms",
    "system design": "system_design",
    "os": "os_notes",
    "operating systems": "os_notes",
    "dbms": "dbms_notes",
    "database": "dbms_notes",
    "networks": "networks_notes",
    "networking": "networks_notes",
    "interview": "interview_qna",
    "best practices": "best_practices",
    "errors": "common_errors",
    "debugging": "common_errors",
}
