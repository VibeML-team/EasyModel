import { useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { v2API } from "../api/v2";
import { useTrainingStore } from "../stores/trainingStore";

export function DashboardPage() {
  const [jobIDInput, setJobIDInput] = useState("");

  const training = useTrainingStore((state) => ({
    jobID: state.jobID,
    setJobID: state.setJobID,
    startPolling: state.startPolling,
    stopPolling: state.stopPolling,
    status: state.status,
    progressPercent: state.progressPercent,
    error: state.error,
  }));

  const pauseMutation = useMutation({
    mutationFn: () => {
      if (!training.jobID) {
        return Promise.reject(new Error("Missing job id"));
      }
      return v2API.pauseTraining(training.jobID);
    },
  });

  const resumeMutation = useMutation({
    mutationFn: () => {
      if (!training.jobID) {
        return Promise.reject(new Error("Missing job id"));
      }
      return v2API.resumeTraining(training.jobID);
    },
  });

  const applyJobID = async (event: FormEvent) => {
    event.preventDefault();
    const value = jobIDInput.trim();
    if (!value) {
      return;
    }
    training.setJobID(value);
    await training.startPolling();
  };

  return (
    <section className="card stack-lg">
      <div>
        <p className="eyebrow">Dashboard scaffold</p>
        <h2 className="section-title">Track V2 training jobs with shared store</h2>
      </div>

      <form className="stack-md" onSubmit={applyJobID}>
        <label className="field-label" htmlFor="job-id-input">
          Job ID
        </label>
        <input
          id="job-id-input"
          className="field"
          placeholder="v2_xxxx"
          value={jobIDInput}
          onChange={(event) => setJobIDInput(event.target.value)}
        />
        <div className="inline-actions">
          <button className="button" type="submit">
            Start polling
          </button>
          <button className="button button-secondary" type="button" onClick={training.stopPolling}>
            Stop polling
          </button>
        </div>
      </form>

      <article className="subcard stack-sm">
        <p>
          <strong>Current job:</strong> {training.jobID || "none"}
        </p>
        <p>
          <strong>Status:</strong> {training.status}
        </p>
        <p>
          <strong>Progress:</strong> {training.progressPercent}%
        </p>
        {training.error && <p className="error">{training.error}</p>}
      </article>

      <div className="inline-actions">
        <button
          className="button button-secondary"
          type="button"
          disabled={!training.jobID || pauseMutation.isPending}
          onClick={() => pauseMutation.mutate()}
        >
          Pause
        </button>
        <button
          className="button button-secondary"
          type="button"
          disabled={!training.jobID || resumeMutation.isPending}
          onClick={() => resumeMutation.mutate()}
        >
          Resume
        </button>
      </div>
    </section>
  );
}
