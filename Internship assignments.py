from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize embedding models
small_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Small model
large_model = SentenceTransformer('nvidia/nv-embedqa-e5-v5')  # Large model

def embed_passages(passages, model):
    """
    Embed the passages using the provided embedding model.
    :param passages: List of passages to embed.
    :param model: SentenceTransformer model for embedding.
    :return: List of embeddings for each passage.
    """
    return model.encode(passages, convert_to_tensor=True)

def retrieve_top_k(query, passages, embeddings, model, k=10):
    """
    Retrieve the top-k most relevant passages based on embeddings similarity.
    :param query: Query text.
    :param passages: List of passages.
    :param embeddings: Precomputed passage embeddings.
    :param model: SentenceTransformer model for query embedding.
    :param k: Number of top results to return.
    :return: List of top-k passages.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_k_indices = np.argsort(scores)[::-1][:k]  # Sort and get top-k
    return [(passages[idx], scores[idx].item()) for idx in top_k_indices]
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize reranking models
small_reranker = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
large_reranker = 'nvidia/nv-rerankqa-mistral-4b-v3'

small_reranker_model = AutoModelForSequenceClassification.from_pretrained(small_reranker)
small_reranker_tokenizer = AutoTokenizer.from_pretrained(small_reranker)

large_reranker_model = AutoModelForSequenceClassification.from_pretrained(large_reranker)
large_reranker_tokenizer = AutoTokenizer.from_pretrained(large_reranker)

def rerank_passages(query, passages, reranker_model, reranker_tokenizer):
    """
    Rerank the top-k passages using a cross-encoder ranking model.
    :param query: Query text.
    :param passages: List of passages to rerank.
    :param reranker_model: Cross-encoder model.
    :param reranker_tokenizer: Tokenizer for the cross-encoder model.
    :return: List of reranked passages.
    """
    inputs = [query + " [SEP] " + passage for passage, _ in passages]
    tokenized_inputs = reranker_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = reranker_model(**tokenized_inputs)
        scores = outputs.logits.squeeze(-1).tolist()

    # Combine passages with scores and rerank
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [(passage[0], score) for passage, score in ranked_passages]
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Initialize reranking models
small_reranker = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
large_reranker = 'nvidia/nv-rerankqa-mistral-4b-v3'

small_reranker_model = AutoModelForSequenceClassification.from_pretrained(small_reranker)
small_reranker_tokenizer = AutoTokenizer.from_pretrained(small_reranker)

large_reranker_model = AutoModelForSequenceClassification.from_pretrained(large_reranker)
large_reranker_tokenizer = AutoTokenizer.from_pretrained(large_reranker)

def rerank_passages(query, passages, reranker_model, reranker_tokenizer):
    """
    Rerank the top-k passages using a cross-encoder ranking model.
    :param query: Query text.
    :param passages: List of passages to rerank.
    :param reranker_model: Cross-encoder model.
    :param reranker_tokenizer: Tokenizer for the cross-encoder model.
    :return: List of reranked passages.
    """
    inputs = [query + " [SEP] " + passage for passage, _ in passages]
    tokenized_inputs = reranker_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = reranker_model(**tokenized_inputs)
        scores = outputs.logits.squeeze(-1).tolist()

    # Combine passages with scores and rerank
    ranked_passages = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [(passage[0], score) for passage, score in ranked_passages]
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def multi_stage_retrieval(query, passages, model_stage1, model_stage2, reranker_model, reranker_tokenizer, k=10):
    """
    Multi-stage retrieval pipeline combining candidate retrieval and reranking.
    :param query: Query text.
    :param passages: List of passages.
    :param model_stage1: Embedding model for candidate retrieval.
    :param model_stage2: Precomputed passage embeddings.
    :param reranker_model: Cross-encoder model for reranking.
    :param reranker_tokenizer: Tokenizer for reranker model.
    :param k: Number of top results to return.
    :return: List of top-k reranked passages.
    """
    # Stage 1: Candidate retrieval
    top_k_passages = retrieve_top_k(query, passages, model_stage2, model_stage1, k)
    
    # Stage 2: Reranking the retrieved passages
    reranked_passages = rerank_passages(query, top_k_passages, reranker_model, reranker_tokenizer)
    
    return reranked_passages

# Example Usage
passages = ["Passage 1 text", "Passage 2 text", "Passage 3 text", "..."]  # Load actual passages here
query = "What is the capital of France?"

# Embed the passages (you can choose to do this once and cache the embeddings)
embeddings = embed_passages(passages, small_model)

# Run the pipeline
top_reranked_passages = multi_stage_retrieval(query, passages, small_model, embeddings, small_reranker_model, small_reranker_tokenizer, k=5)
for passage, score in top_reranked_passages:
    print(f"Passage: {passage} - Score: {score}")
from sklearn.metrics import ndcg_score
import numpy as np

def calculate_ndcg(relevance_scores, predicted_ranks, k=10):
    """
    Calculate NDCG@k for the predicted results.
    :param relevance_scores: Ground truth relevance scores.
    :param predicted_ranks: Predicted relevance scores.
    :param k: Number of results to consider for NDCG.
    :return: NDCG score.
    """
    true_relevance = np.asarray([relevance_scores])
    predicted_relevance = np.asarray([predicted_ranks])
    return ndcg_score(true_relevance, predicted_relevance, k=k)
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

# Download Natural Questions dataset (NQ) from BEIR
dataset = "natural-questions"  # Change this to "hotpotqa" or "fiqa" for other datasets
data_path = util.download_and_unzip(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip", "./datasets")
# Load the dataset using the BEIR data loader
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")  # Test set; can also load "train" or "dev"
# Load the dataset using the BEIR data loader
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")  # Test set; can also load "train" or "dev"
def chunk_document(doc_text, max_chunk_size=200, overlap_size=50):
    """
    Split a document into smaller passages based on a max_chunk_size.
    :param doc_text: The full document text.
    :param max_chunk_size: The maximum number of tokens per chunk.
    :param overlap_size: Number of tokens overlapping between chunks (for context preservation).
    :return: List of document chunks.
    """
    words = doc_text.split()
    chunks = []
    
    for i in range(0, len(words), max_chunk_size - overlap_size):
        chunk = words[i:i + max_chunk_size]
        chunks.append(" ".join(chunk))
    
    return chunks

# Example of chunking documents in the corpus
passages = []
passage_id = 0
for doc_id, doc in corpus.items():
    doc_text = doc["text"]
    doc_chunks = chunk_document(doc_text)
    
    # Add each chunk as a separate passage
    for chunk in doc_chunks:
        passages.append({
            "doc_id": f"{doc_id}_{passage_id}",  # Assign unique passage ID
            "text": chunk
        })
        passage_id += 1

print(f"Total number of passages: {len(passages)}")
from transformers import AutoTokenizer

# Example: Use the tokenizer for your selected embedding model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize passages
tokenized_passages = [tokenizer(p['text'], truncation=True, padding=True, return_tensors="pt") for p in passages]

# Tokenize queries
tokenized_queries = [tokenizer(q, truncation=True, padding=True, return_tensors="pt") for q in queries.values()]
from transformers import AutoTokenizer

# Example: Use the tokenizer for your selected embedding model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize passages
tokenized_passages = [tokenizer(p['text'], truncation=True, padding=True, return_tensors="pt") for p in passages]

# Tokenize queries
tokenized_queries = [tokenizer(q, truncation=True, padding=True, return_tensors="pt") for q in queries.values()]
import json

# Save tokenized passages and queries to disk for future use
with open('preprocessed_passages.json', 'w') as f:
    json.dump(passages, f)

with open('preprocessed_queries.json', 'w') as f:
    json.dump(queries, f)

# To reload later:
# with open('preprocessed_passages.json') as f:
#     passages = json.load(f)
# with open('preprocessed_queries.json') as f:
#     queries = json.load(f)
