import { useState } from "react";
import type { FormEvent } from "react";
import "./App.css";

type SearchHit = {
  line: string;
  score: number;
  cluster_index: string;
  rerank_score?: number | null;
};

type SearchResponse = {
  query: string;
  clusters: string[];
  results: SearchHit[];
};

// TD2-6 prompt: sent to the endpoint http://localhost:8000/search
const SEARCH_URL = "http://localhost:8000/search";

function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<SearchResponse | null>(null);

  async function onSearch(event: FormEvent) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(SEARCH_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          top_clusters: 3,
          top_docs: 10,
          use_rerank: true,
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `HTTP ${response.status}`);
      }

      const payload = (await response.json()) as SearchResponse;
      setData(payload);
    } catch (err) {
      setData(null);
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="brand">IASearch</p>
        <h1>Recherche sémantique</h1>
        <p className="subtitle">
          KNN clusters → recherche hybride BM25+KNN → rerank LLM
        </p>
      </header>

      {/* TD2-6: text input + button "Search" */}
      <form className="search" onSubmit={onSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Recherche..."
          aria-label="Recherche"
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "..." : "Search"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {data && (
        <section className="results">
          <p className="meta">
            Clusters : {data.clusters.join(", ") || "aucun"} —{" "}
            {data.results.length} résultat(s)
          </p>
          <ul>
            {data.results.map((hit) => (
              <li key={`${hit.cluster_index}-${hit.line}`}>
                <p className="line">{hit.line}</p>
                <p className="scores">
                  <span>{hit.cluster_index}</span>
                  <span>ES {hit.score.toFixed(3)}</span>
                  {hit.rerank_score != null && (
                    <span>Rerank {hit.rerank_score.toFixed(3)}</span>
                  )}
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

export default App;
