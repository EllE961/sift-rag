# Sift — common tasks. On Windows run these from Git Bash, or copy the commands.
PY ?= python

.PHONY: help data index serve test lint eval-retrieval eval-blend eval-hnsw eval-claims eval-claims-claude projection figures all-evals

help:          ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

data:          ## download BEIR SciFact + NFCorpus and the SciFact claim release (~8 MB)
	$(PY) -m sift.datasets download

projection:    ## build collections (BM25 + bge-small + HNSW) and the 2-D embedding maps
	$(PY) -m sift.eval projection

serve:         ## run the API + web UI on http://127.0.0.1:8002
	$(PY) -m sift.server

test:          ## unit + integration tests (CPU, no network, ~30 s)
	$(PY) -m pytest

lint:          ## ruff
	ruff check .

eval-retrieval: ## BEIR ablation: BM25 / dense / hybrid / rerank (writes results/retrieval.json)
	$(PY) -m sift.eval retrieval

eval-blend:    ## tune the cross-encoder blend on train, report on test
	$(PY) -m sift.eval blend

eval-hnsw:     ## HNSW recall/QPS benchmark on ~79k sentence embeddings
	$(PY) -m sift.eval hnsw

eval-claims:   ## end-to-end SciFact claim verification, offline NLI verifier
	$(PY) -m sift.eval claims --verifier nli

eval-claims-claude: ## same with Claude (needs ANTHROPIC_API_KEY; ~99 claims x 4 conditions)
	$(PY) -m sift.eval claims --verifier claude

figures:       ## render docs/assets/*.png from results/*.json
	$(PY) -m sift.eval figures

all-evals: eval-retrieval eval-blend eval-hnsw eval-claims figures
