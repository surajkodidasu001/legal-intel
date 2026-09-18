"""Sanity check: can we actually create a table, insert a vector, and run a
similarity query against the pgvector-enabled Postgres container?

Run from the host (uses the exposed 5432 port, not the in-network 'db' hostname):
    python test_pgvector.py
"""
import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://legal:legal@localhost:5432/legalintel"

conn = psycopg.connect(DSN, autocommit=True)
register_vector(conn)
cur = conn.cursor()

print("1. Dropping old test table if present...")
cur.execute("DROP TABLE IF EXISTS embedding_test;")

print("2. Creating table with a 4-dim vector column...")
cur.execute("""
    CREATE TABLE embedding_test (
        id SERIAL PRIMARY KEY,
        label TEXT,
        embedding VECTOR(4)
    );
""")

print("3. Inserting three fake embeddings...")
rows = [
    ("contract clause A", [1.0, 0.0, 0.0, 0.0]),
    ("contract clause B", [0.9, 0.1, 0.0, 0.0]),   # close to A
    ("contract clause C", [0.0, 0.0, 1.0, 0.0]),   # far from A
]
for label, vec in rows:
    cur.execute(
        "INSERT INTO embedding_test (label, embedding) VALUES (%s, %s)",
        (label, vec),
    )

print("4. Querying nearest neighbors to [1.0, 0.0, 0.0, 0.0]...")
cur.execute("""
    SELECT label, embedding <-> %s::vector AS distance
    FROM embedding_test
    ORDER BY distance
    LIMIT 5;
""", ([1.0, 0.0, 0.0, 0.0],))

for label, distance in cur.fetchall():
    print(f"   {label}: distance={distance:.4f}")

print("5. Cleaning up...")
cur.execute("DROP TABLE embedding_test;")

print("\nDone. If clause A and B came back as the two nearest (in that order,")
print("both much closer than C), pgvector similarity search is working correctly.")

conn.close()
