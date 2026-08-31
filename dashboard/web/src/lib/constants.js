export const DOMAIN_HEX = {
  finance: '#047857', hr: '#A21CAF', legal: '#B45309', engineering: '#0369A1',
}

// Order matches pipeline/graph.py's build_graph() wiring.
export const NODE_ORDER = [
  'abac_gate', 'domain_router', 'query_analyzer', 'hybrid_retrieval',
  'reranker', 'crag_grader', 'graphrag_augmenter', 'generator', 'hallucination_check',
]
export const REFUSAL_PATH = ['abac_gate', 'domain_router', 'refusal']

export const NODE_HEX = {
  abac_gate: '#8A93A3', domain_router: '#0369A1', query_analyzer: '#0369A1',
  hybrid_retrieval: '#047857', reranker: '#047857', crag_grader: '#B45309',
  graphrag_augmenter: '#A21CAF', generator: '#4338CA', hallucination_check: '#4338CA',
  refusal: '#B91C1C',
}

export const TYPE_HEX = {
  Employee: '#0369A1', Project: '#047857', Incident: '#B91C1C', System: '#B45309',
  Budget: '#A21CAF', Contract: '#DB2777', Vendor: '#0D9488', Department: '#64748B',
  Review: '#CA8A04',
}

export const gradeClass = (g) =>
  g === 'correct' ? 't-ok' : g === 'ambiguous' ? 't-wa' : g === 'incorrect' ? 't-ba' : 't-nu'
