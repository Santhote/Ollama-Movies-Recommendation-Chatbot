import ollama
import re
import concurrent.futures
from qdrant_client import QdrantClient
from openai import OpenAI
import re
import ollama
from qdrant_client import QdrantClient
from openai import OpenAI

# Clients 

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

qdrant_client = QdrantClient(url="http://localhost:6333")

# Helpers

def extract_request(user_text: str):
    match = re.search(r"(\d+)\s+movies?", user_text.lower())
    k = int(match.group(1)) if match else 5

    query = re.sub(r"\d+\s+movies?", "", user_text, flags=re.I)
    query = re.sub(r"recommend|like|similar|movies", "", query, flags=re.I)
    query = query.strip()

    return query, k


def get_embedding(text, max_chars=400):
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars]

    return ollama.embeddings(
        model="mxbai-embed-large",
        prompt=text
    )["embedding"]


def similarity_search(query, limit):
    vector = get_embedding(query)
    result = qdrant_client.query_points(
        collection_name="movies",
        query=vector,
        limit=limit
    )
    return result.points or []


def parse_movie(point):
    p = point.payload
    return {
        "title": p.get("title", "").strip(),
        "overview": p.get("overview", "")[:120]
    }

# Main Function

def run_conversation(user_query: str):
    query, k = extract_request(user_query)
    query_lower = query.lower()

    # Step 1: Retrieve more movies than needed
    points = similarity_search(query, limit=k * 6)

    # Step 2: Deduplicate & strictly select exact k movies
    movies = []
    seen = set()

    for p in points:
        movie = parse_movie(p)
        title_lower = movie["title"].lower()

        if title_lower == query_lower:
            continue

        if title_lower not in seen:
            seen.add(title_lower)
            movies.append(movie)

        if len(movies) == k:
            break

    # Safety fallback
    movies = movies[:k]

    # Step 3: Ask LLM to generate reasons only
    movie_list_text = "\n".join(
        [f"{i+1}. {m['title']}" for i, m in enumerate(movies)]
    )

    response = ollama_client.chat.completions.create(
        model="qwen:7b",
        temperature=0.3,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a movie recommendation expert.\n"
                    "For EACH movie, write exactly ONE clear reason.\n"
                    "Do NOT add or remove movies.\n"
                    "Return the same numbered list."
                )
            },
            {
                "role": "user",
                "content": (
                    f"I like: {query}\n\n"
                    f"Movies:\n{movie_list_text}\n\n"
                    "Add a reason for each movie."
                )
            }
        ]
    )

    return response.choices[0].message.content
# Clients
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

qdrant_client = QdrantClient(url="http://localhost:6333")


# Extract count + query
def extract_request(user_text: str):
    match = re.search(r"(\d+)\s+movies?", user_text.lower())
    k = int(match.group(1)) if match else 5

    clean_query = re.sub(r"\d+\s+movies?", "", user_text, flags=re.I)
    clean_query = re.sub(r"recommend|like|similar|movies", "", clean_query, flags=re.I)
    clean_query = clean_query.strip()

    return clean_query, k


# Embedding
def get_embedding(text, max_chars=400):
    text = str(text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]

    return ollama.embeddings(
        model="mxbai-embed-large",
        prompt=text
    )["embedding"]


# Retrieval
def similarity_search(query, limit):
    vector = get_embedding(query)
    result = qdrant_client.query_points(
        collection_name="movies",
        query=vector,
        limit=limit
    )
    return result.points or []


# Context formatting
def parse_similar_context(point):
    p = point.payload
    return {
        "title": p.get("title", ""),
        "genre": p.get("genre", ""),
        "overview": p.get("overview", "")[:160]
    }


# Main RAG pipeline
def run_conversation(user_query):
    query, k = extract_request(user_query)
    query_lower = query.lower()

    # retrieve more than needed
    points = similarity_search(query, limit=max(20, k * 4))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        raw_context = list(executor.map(parse_similar_context, points))

    # Only remove exact same movie title
    filtered = []
    for item in raw_context:
        if item["title"].lower() == query_lower:
            continue
        if item["title"] not in [m["title"] for m in filtered]:
            filtered.append(item)

    context = filtered[: max(12, k * 2)]

    response = ollama_client.chat.completions.create(
        model="qwen:4b",
        temperature=0.55,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a movie recommendation expert.\n"
                    f"Recommend EXACTLY {k} movies.\n"
                    "You MAY recommend sequels or related films.\n"
                    "Do NOT repeat the exact same movie title.\n"
                    "Each movie MUST have a UNIQUE, specific reason.\n"
                    "Finish the full list.\n"
                    "Output as a numbered list with title and reason."
                )
            },
            {
                "role": "user",
                "content": (
                    f"I like: {query}\n\n"
                    "Candidate movies:\n"
                    + "\n".join(
                        [
                            f"- {c['title']} ({c['genre']}): {c['overview']}"
                            for c in context
                        ]
                    )
                    + f"\n\nRecommend {k} similar movies."
                )
            }
        ]
    )

    return response.choices[0].message.content
