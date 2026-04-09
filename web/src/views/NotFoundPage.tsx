import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <section className="card stack-md">
      <p className="eyebrow">404</p>
      <h2 className="section-title">Route not found</h2>
      <p className="muted">This route has not been migrated yet.</p>
      <Link className="action-link" to="/">
        Back to home
      </Link>
    </section>
  );
}
