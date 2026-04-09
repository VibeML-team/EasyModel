import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { v1API } from "../api/v1";

export function HomePage() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: v1API.health,
  });

  return (
    <section className="card stack-lg">
      <div>
        <p className="eyebrow">Phase 1</p>
        <h2 className="section-title">Frontend foundation is now in React</h2>
        <p className="muted">
          This app is a migration shell. Existing legacy pages remain available while V2 and dashboard
          logic are gradually moved to typed modules.
        </p>
      </div>

      <div className="grid-two">
        <article className="subcard">
          <h3>Backend status</h3>
          {healthQuery.isLoading && <p className="muted">Loading health...</p>}
          {healthQuery.error && <p className="error">Failed to load health.</p>}
          {healthQuery.data && (
            <ul className="kv-list">
              <li>
                <span>Status</span>
                <strong>{healthQuery.data.status}</strong>
              </li>
              <li>
                <span>Version</span>
                <strong>{healthQuery.data.version || "unknown"}</strong>
              </li>
              <li>
                <span>LLM</span>
                <strong>{healthQuery.data.llm_model || "fallback mode"}</strong>
              </li>
            </ul>
          )}
        </article>

        <article className="subcard stack-md">
          <h3>Migration routes</h3>
          <Link className="action-link" to="/v2">
            Open V2 migration page
          </Link>
          <Link className="action-link" to="/dashboard">
            Open Dashboard migration page
          </Link>
          <a className="action-link" href="/chat">
            Keep using legacy chat page
          </a>
        </article>
      </div>
    </section>
  );
}
