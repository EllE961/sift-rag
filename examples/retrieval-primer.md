# A field guide to retrieval for RAG

Retrieval-augmented generation lives or dies on its first stage. A language model can only cite what it is shown, so the job of retrieval is to put the right evidence in front of it, cheaply, every time.

## Lexical retrieval

BM25 scores a document by how often it contains each query term, with two corrections. Term frequency saturates: the tenth occurrence of a word adds far less than the first, controlled by the parameter k1. Long documents are normalized against the average length, controlled by b. Rare terms matter more than common ones through the inverse document frequency.

BM25 needs no training data, runs in microseconds over an inverted index and is remarkably hard to beat on keyword-heavy domains such as biomedicine. Its weakness is vocabulary mismatch: a query about "heart attacks" will not match a passage about "myocardial infarction".

## Dense retrieval

A bi-encoder maps queries and passages into the same vector space, so that texts with similar meaning land close together. Because passages are embedded independently of the query, the whole corpus can be encoded once, offline. At query time the problem reduces to nearest-neighbor search.

Dense retrievers fix vocabulary mismatch but introduce new failure modes. They can be confidently wrong on rare entities, numbers and exact identifiers, and they inherit the domain of their training data.

## Approximate nearest neighbors

Exact search compares the query with every vector, which is linear in the corpus size. Hierarchical navigable small world graphs trade a little recall for orders of magnitude in speed. Every vector is a node in a proximity graph; a few nodes are promoted to sparser upper layers, like the express lanes of a skip list. Search descends greedily through the upper layers and then runs a beam search on the bottom layer.

The beam width ef is the knob. A wider beam explores more of the graph, raising recall and lowering throughput. Queries that fall far from the data, such as short questions searched against long passages, need a wider beam than queries that look like the corpus.

## Fusion

Lexical and dense retrievers make different mistakes, which is exactly why combining them works. Reciprocal rank fusion ignores raw scores and adds one over k plus the rank for each list a document appears in. It needs no calibration and is a strong default. A convex combination of normalized scores can do better, but its weight must be tuned on held-out queries, never on the test set.

## Reranking

A cross-encoder reads the query and a candidate passage together, so every query token can attend to every passage token. It is far more accurate than a bi-encoder per pair and far too slow to run over a whole corpus, so it reorders only the top candidates.

A reranker is only as good as its training domain. A model trained on web search can rank scientific abstracts worse than the hybrid first stage it is meant to improve. Blending its score with the first-stage score, instead of replacing one with the other, often recovers the gain.

## Grounded generation

The final step hands the top passages to a language model with instructions to answer only from them. Citations that point to exact character spans make every claim checkable, and a system that can say "the documents do not contain the answer" is worth more than one that always answers.
