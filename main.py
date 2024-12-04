import numpy as np
import re

def split_text_into_tokens(text):
    return [word for word in re.split(r'[,\.\?!: ]+', text.lower()) if word]

def build_incidence_matrix(doc_tokens, unique_tokens):
    return np.array([[1 if token in doc else 0 for doc in doc_tokens] for token in unique_tokens])

def process_query(query, unique_tokens):
    return np.array([1 if token in query else 0 for token in unique_tokens])

def calculate_similarity(query_vec, reduced_doc_vecs):
    similarities = []
    query_norm = np.linalg.norm(query_vec)
    for doc_vec in reduced_doc_vecs:
        doc_norm = np.linalg.norm(doc_vec)
        similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
        similarities.append(round(float(similarity), 2))
    return similarities

num_docs = int(input())
documents = [input().strip() for _ in range(num_docs)]
query = input().strip()
reduction_rank = int(input())

document_tokens = [split_text_into_tokens(doc) for doc in documents]
query_tokens = split_text_into_tokens(query)
unique_words = set(word for tokens in document_tokens for word in tokens)
incidence_matrix = build_incidence_matrix(document_tokens, unique_words)

U, S, Vt = np.linalg.svd(incidence_matrix.T, full_matrices=False)
U_reduced = U[:, :reduction_rank]
S_reduced = np.diag(S[:reduction_rank])
Vt_reduced = Vt[:reduction_rank, :]

query_vector = process_query(query_tokens, unique_words)
query_in_reduced_space = query_vector @ Vt_reduced.T @ np.linalg.inv(S_reduced)
reduced_document_vectors = U_reduced @ S_reduced
similarities = calculate_similarity(query_in_reduced_space, reduced_document_vectors)

print(similarities)