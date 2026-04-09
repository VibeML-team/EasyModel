import type { V2CheckpointCandidate } from "../../types";

type V2CheckpointListProps = {
  title: string;
  checkpoints: V2CheckpointCandidate[];
  selectedID?: string | null;
  onSelect?: (checkpointID: string) => void;
  onFeedback?: (checkpointID: string, liked: boolean) => void;
  downloadURLBuilder?: (checkpointID: string) => string;
};

export function V2CheckpointList({
  title,
  checkpoints,
  selectedID,
  onSelect,
  onFeedback,
  downloadURLBuilder,
}: V2CheckpointListProps) {
  return (
    <section className="subcard stack-md">
      <h3>{title}</h3>
      {checkpoints.length === 0 ? (
        <p className="muted">No checkpoints yet.</p>
      ) : (
        <div className="checkpoint-grid">
          {checkpoints.map((checkpoint, index) => {
            const isSelected = selectedID === checkpoint.id;
            const className = isSelected ? "checkpoint-card checkpoint-card-selected" : "checkpoint-card";
            return (
              <article key={checkpoint.id} className={className}>
                <button
                  type="button"
                  className="checkpoint-main"
                  onClick={() => onSelect?.(checkpoint.id)}
                  disabled={!onSelect}
                >
                  <div className="checkpoint-top">
                    <span className="badge">#{index + 1}</span>
                    <strong>{(checkpoint.metric * 100).toFixed(2)}%</strong>
                  </div>
                  <p className="muted">
                    Step {checkpoint.step} · {checkpoint.metric_name || "metric"}
                  </p>
                  {checkpoint.model ? <p className="muted">{checkpoint.model}</p> : null}
                </button>

                <div className="inline-actions">
                  {onFeedback ? (
                    <>
                      <button type="button" className="button button-secondary" onClick={() => onFeedback(checkpoint.id, true)}>
                        Like
                      </button>
                      <button type="button" className="button button-secondary" onClick={() => onFeedback(checkpoint.id, false)}>
                        Dislike
                      </button>
                    </>
                  ) : null}

                  {downloadURLBuilder ? (
                    <a className="action-link" href={downloadURLBuilder(checkpoint.id)}>
                      Download
                    </a>
                  ) : null}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
