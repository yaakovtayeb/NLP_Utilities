# bin/elasticsearch -E xpack.security.enabled=false
import json
import requests
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes import TfidfRetriever, BM25Retriever


document_store = ElasticsearchDocumentStore(
    host='localhost', username='ytayeb1', password='ytayeb1_pass', index='squad_docs'
)

res = requests.get('http://localhost:9200/_cluster/health')
# count index
requests.get('http://localhost:9200/squad_docs/_count').json()

# remove index from elastic search
res = requests.post('http://localhost:9200/squad_docs/_delete_by_query',
                    json = {
                        'query': {
                            'match_all': {}
                        }
                    })

# read json
with open('tests/squad_data/dev.json', 'r') as f:
    squad = json.load(f)

squad_docs = []
for sample in squad:
    squad_docs.append({
        'content': sample['context']
    })

# store information in elastic search
document_store.write_documents(squad_docs)

retriever = TfidfRetriever(document_store)
retriever = BM25Retriever(document_store)
retriever.retrieve("Physics is a very abstract subject")[0]




