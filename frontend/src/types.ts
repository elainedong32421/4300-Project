export interface AitaPost {
  id: number;
  submission_id: string;
  title: string;
  selftext: string;
  score: number;
  similarity: number;
  verdict?: string;
  /** Present for SVD retrieval only; ignored in TF-IDF and in the RAG UI. */
  svd_top_dimensions?: SvdDimensionValue[];
  tfidf_similarity?: number;
  original_rank?: number;
  rerank_reason?: string;
}

export interface SvdDimensionValue {
  dimension: number;
  label: string;
  post_value: number;
  query_value: number;
  contribution: number;
}

export interface LlmSearchResponse {
  rewritten_query: string;
  ir_results: AitaPost[];
  reranked_results?: AitaPost[];
  llm_answer: string;
  verdict_filter: string | null;
}
