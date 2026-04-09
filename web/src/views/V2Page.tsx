import { useEffect, useMemo, useState, type FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { v2API } from "../api/v2";
import { V2CheckpointList } from "../components/v2/V2CheckpointList";
import { V2ClarificationQuestions } from "../components/v2/V2ClarificationQuestions";
import { V2FlowStepper } from "../components/v2/V2FlowStepper";
import { useSessionStore } from "../stores/sessionStore";
import { useTrainingStore } from "../stores/trainingStore";
import type { V2CheckpointCandidate, V2CompileResponse, V2Domain } from "../types";

const domainOptions: V2Domain[] = ["general", "ai4science", "gnn", "timeseries", "rl"];

const flowSteps = [
  { id: 1, title: "Compile" },
  { id: 2, title: "Review" },
  { id: 3, title: "Train" },
  { id: 4, title: "Finalize" },
];

type TrainingAlert = {
  type: string;
  message: string;
};

type TrainingObservation = {
  progress?: {
    stage?: string;
  };
  alerts?: TrainingAlert[];
};

function sortByMetric(items: V2CheckpointCandidate[]) {
  return [...items].sort((a, b) => b.metric - a.metric);
}

export function V2Page() {
  const [intent, setIntent] = useState("Predict customer churn with higher recall priority.");
  const [domain, setDomain] = useState<V2Domain>("general");
  const [currentStep, setCurrentStep] = useState(1);
  const [compiled, setCompiled] = useState<V2CompileResponse | null>(null);
  const [clarificationAnswers, setClarificationAnswers] = useState<Record<string, string>>({});
  const [clarificationQuestions, setClarificationQuestions] = useState<
    { id: string; text: string; type: string; options?: string[]; context?: string; priority?: number }[]
  >([]);
  const [interimCheckpoints, setInterimCheckpoints] = useState<V2CheckpointCandidate[]>([]);
  const [finalCandidates, setFinalCandidates] = useState<V2CheckpointCandidate[]>([]);
  const [selectedCheckpointID, setSelectedCheckpointID] = useState<string | null>(null);
  const [selectionMessage, setSelectionMessage] = useState<string | null>(null);

  const sessionID = useSessionStore((state) => state.sessionID);
  const setSessionID = useSessionStore((state) => state.setSessionID);

  const jobID = useTrainingStore((state) => state.jobID);
  const setJobID = useTrainingStore((state) => state.setJobID);
  const startPolling = useTrainingStore((state) => state.startPolling);
  const stopPolling = useTrainingStore((state) => state.stopPolling);
  const trainingStatus = useTrainingStore((state) => state.status);
  const progressPercent = useTrainingStore((state) => state.progressPercent);
  const observation = useTrainingStore((state) => state.observation);
  const trainingError = useTrainingStore((state) => state.error);

  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  const compileMutation = useMutation({
    mutationFn: v2API.compile,
    onSuccess: (result) => {
      setSessionID(result.session_id);
      setCompiled(result);
      setClarificationAnswers({});
      setCurrentStep(1);
      if (result.intent_card.clarification.is_ambiguous) {
        setClarificationQuestions(result.intent_card.clarification.suggested_questions);
      } else {
        setClarificationQuestions([]);
      }
    },
  });

  const clarifyMutation = useMutation({
    mutationFn: v2API.clarify,
    onSuccess: (result) => {
      setCompiled((previous) =>
        previous
          ? {
              ...previous,
              intent_card: result.intent_card,
              can_proceed: result.clarification_complete,
            }
          : previous,
      );
      setClarificationQuestions(result.remaining_questions);
      if (result.clarification_complete) {
        setCurrentStep(2);
      }
    },
  });

  const startTrainingMutation = useMutation({
    mutationFn: v2API.startTraining,
    onSuccess: async (result) => {
      setJobID(result.job_id);
      await startPolling();
      setCurrentStep(3);
      setInterimCheckpoints([]);
      setFinalCandidates([]);
      setSelectedCheckpointID(null);
      setSelectionMessage(null);
    },
  });

  const listCheckpointsMutation = useMutation({
    mutationFn: v2API.listCheckpoints,
    onSuccess: (result) => {
      setInterimCheckpoints(sortByMetric(result.checkpoints));
    },
  });

  const pauseMutation = useMutation({ mutationFn: v2API.pauseTraining });
  const resumeMutation = useMutation({ mutationFn: v2API.resumeTraining });

  const pauseAndDownloadMutation = useMutation({
    mutationFn: v2API.pauseAndDownload,
    onSuccess: (result) => {
      setInterimCheckpoints(sortByMetric(result.checkpoints));
    },
  });

  const finalizeMutation = useMutation({
    mutationFn: v2API.finalizeTraining,
    onSuccess: (result) => {
      const candidates = sortByMetric(result.acceptance_card.candidates || []);
      setFinalCandidates(candidates);
      setSelectedCheckpointID(candidates[0]?.id || null);
      setCurrentStep(4);
    },
  });

  const feedbackMutation = useMutation({ mutationFn: v2API.feedback });

  const selectVersionMutation = useMutation({
    mutationFn: v2API.selectVersion,
    onSuccess: (result) => {
      setSelectionMessage(`Selected ${result.selected_checkpoint}. Model path: ${result.model_path}`);
    },
  });

  const submitCompile = (event: FormEvent) => {
    event.preventDefault();
    if (!intent.trim()) {
      return;
    }
    compileMutation.mutate({ intent: intent.trim(), domain });
  };

  const canProceed = useMemo(() => {
    if (!compiled) {
      return false;
    }
    return compiled.can_proceed;
  }, [compiled]);

  const canSubmitClarification = useMemo(() => {
    if (clarificationQuestions.length === 0) {
      return false;
    }
    return clarificationQuestions.every((question) => Boolean(clarificationAnswers[question.id]?.trim()));
  }, [clarificationAnswers, clarificationQuestions]);

  const typedObservation = observation as TrainingObservation | null;
  const progressStage = typedObservation?.progress?.stage || "training";
  const alerts = typedObservation?.alerts || [];
  const canGenerateFinalCandidates = Boolean(jobID) && (progressPercent >= 100 || trainingStatus === "idle");

  const handleSubmitClarification = () => {
    if (!sessionID) {
      return;
    }
    const answers = clarificationQuestions
      .map((question) => ({
        question_id: question.id,
        answer: clarificationAnswers[question.id]?.trim() || "",
      }))
      .filter((item) => item.answer.length > 0);

    clarifyMutation.mutate({ session_id: sessionID, answers });
  };

  const handlePauseTraining = async () => {
    if (!jobID) {
      return;
    }
    await pauseMutation.mutateAsync(jobID);
    await listCheckpointsMutation.mutateAsync(jobID);
  };

  const handleResumeTraining = async () => {
    if (!jobID) {
      return;
    }
    await resumeMutation.mutateAsync(jobID);
    await startPolling();
  };

  const handlePauseAndDownload = async () => {
    if (!jobID) {
      return;
    }
    await pauseAndDownloadMutation.mutateAsync(jobID);
  };

  const handleGenerateCandidates = async () => {
    if (!jobID) {
      return;
    }
    await finalizeMutation.mutateAsync(jobID);
  };

  const handleCheckpointFeedback = async (checkpointID: string, liked: boolean) => {
    if (!jobID) {
      return;
    }
    await feedbackMutation.mutateAsync({
      job_id: jobID,
      checkpoint_id: checkpointID,
      liked,
    });
  };

  const handleConfirmSelection = async () => {
    if (!jobID || !selectedCheckpointID) {
      return;
    }
    await selectVersionMutation.mutateAsync({
      job_id: jobID,
      checkpoint_id: selectedCheckpointID,
    });
  };

  return (
    <section className="card stack-lg">
      <div>
        <p className="eyebrow">V2 flow migration</p>
        <h2 className="section-title">Intent compile, clarify, train, and finalize</h2>
        <p className="muted">This page now implements the full V2 flow with typed API integration.</p>
      </div>

      <V2FlowStepper steps={flowSteps} currentStep={currentStep} />

      {currentStep === 1 ? (
        <>
          <form className="stack-md" onSubmit={submitCompile}>
            <label className="field-label" htmlFor="intent">
              Intent
            </label>
            <textarea
              id="intent"
              className="field"
              rows={5}
              value={intent}
              onChange={(event) => setIntent(event.target.value)}
            />

            <label className="field-label" htmlFor="domain">
              Domain
            </label>
            <select
              id="domain"
              className="field"
              value={domain}
              onChange={(event) => setDomain(event.target.value as V2Domain)}
            >
              {domainOptions.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>

            <div className="inline-actions">
              <button className="button" type="submit" disabled={compileMutation.isPending}>
                {compileMutation.isPending ? "Compiling..." : "Compile intent"}
              </button>
            </div>
          </form>

          {compileMutation.error ? <p className="error">Compile request failed.</p> : null}

          {compiled ? (
            <article className="subcard stack-sm">
              <h3>System understanding</h3>
              <p>
                <strong>Session:</strong> {compiled.session_id}
              </p>
              <p>
                <strong>Task:</strong> {compiled.intent_card.task.type} ({Math.round(compiled.intent_card.task.confidence * 100)}%)
              </p>
              <p>
                <strong>Reasoning:</strong> {compiled.intent_card.system_notes.reasoning}
              </p>
              <p>
                <strong>Can proceed:</strong> {String(compiled.can_proceed)}
              </p>
              <p className="muted">{compiled.summary}</p>

              {compiled.intent_card.clarification.is_ambiguous ? (
                <div className="stack-sm">
                  <p className="error">
                    Ambiguity score: {(compiled.intent_card.clarification.ambiguity_score * 100).toFixed(0)}%
                  </p>
                  {compiled.intent_card.clarification.missing_elements.length ? (
                    <p className="muted">
                      Missing: {compiled.intent_card.clarification.missing_elements.join(", ")}
                    </p>
                  ) : null}
                </div>
              ) : null}
            </article>
          ) : null}

          <V2ClarificationQuestions
            questions={clarificationQuestions}
            answers={clarificationAnswers}
            onAnswerChange={(questionID, value) => {
              setClarificationAnswers((previous) => ({
                ...previous,
                [questionID]: value,
              }));
            }}
          />

          <div className="inline-actions">
            {clarificationQuestions.length ? (
              <button
                className="button button-secondary"
                type="button"
                disabled={!canSubmitClarification || clarifyMutation.isPending}
                onClick={handleSubmitClarification}
              >
                {clarifyMutation.isPending ? "Submitting..." : "Submit clarification"}
              </button>
            ) : null}

            <button className="button" type="button" disabled={!canProceed} onClick={() => setCurrentStep(2)}>
              Continue to review
            </button>
          </div>
        </>
      ) : null}

      {currentStep === 2 ? (
        <>
          <section className="grid-two">
            <article className="subcard stack-sm">
              <h3>Intent card</h3>
              <p className="muted">Must keep: {compiled?.intent_card.constraints.must_keep.join(", ") || "none"}</p>
              <p className="muted">Can change: {compiled?.intent_card.constraints.can_change.join(", ") || "none"}</p>
              <p className="muted">Priority: {compiled?.intent_card.constraints.priority || "unknown"}</p>
            </article>

            <article className="subcard stack-sm">
              <h3>Plan card</h3>
              <p className="muted">Approach: {compiled?.plan_card.approach.name || "unknown"}</p>
              <p className="muted">Reason: {compiled?.plan_card.approach.reasoning || "n/a"}</p>
              <p className="muted">Primary goal: {compiled?.plan_card.goals.primary || "n/a"}</p>
            </article>
          </section>

          <div className="inline-actions">
            <button className="button button-secondary" type="button" onClick={() => setCurrentStep(1)}>
              Back
            </button>
            <button
              className="button"
              type="button"
              disabled={!sessionID || startTrainingMutation.isPending}
              onClick={() => {
                if (!sessionID) {
                  return;
                }
                startTrainingMutation.mutate({ session_id: sessionID });
              }}
            >
              {startTrainingMutation.isPending ? "Starting..." : "Start training"}
            </button>
          </div>
        </>
      ) : null}

      {currentStep === 3 ? (
        <>
          <article className="subcard stack-sm">
            <h3>Training status</h3>
            <p>
              <strong>Job:</strong> {jobID || "n/a"}
            </p>
            <p>
              <strong>Status:</strong> {trainingStatus}
            </p>
            <p>
              <strong>Stage:</strong> {progressStage}
            </p>
            <p>
              <strong>Progress:</strong> {progressPercent}%
            </p>
            {trainingError ? <p className="error">{trainingError}</p> : null}
            {alerts.length ? (
              <div className="stack-sm">
                {alerts.map((alert, index) => (
                  <p key={`${alert.message}-${index}`} className="error">
                    {alert.type}: {alert.message}
                  </p>
                ))}
              </div>
            ) : null}
          </article>

          <div className="inline-actions">
            <button
              className="button button-secondary"
              type="button"
              disabled={!jobID || pauseMutation.isPending}
              onClick={() => void handlePauseTraining()}
            >
              Pause
            </button>
            <button
              className="button button-secondary"
              type="button"
              disabled={!jobID || resumeMutation.isPending}
              onClick={() => void handleResumeTraining()}
            >
              Resume
            </button>
            <button
              className="button button-secondary"
              type="button"
              disabled={!jobID || listCheckpointsMutation.isPending}
              onClick={() => {
                if (!jobID) {
                  return;
                }
                listCheckpointsMutation.mutate(jobID);
              }}
            >
              Refresh checkpoints
            </button>
            <button
              className="button button-secondary"
              type="button"
              disabled={!jobID || pauseAndDownloadMutation.isPending}
              onClick={() => void handlePauseAndDownload()}
            >
              Pause + load checkpoints
            </button>
            <button
              className="button"
              type="button"
              disabled={!canGenerateFinalCandidates || finalizeMutation.isPending}
              onClick={() => void handleGenerateCandidates()}
            >
              Generate candidates
            </button>
          </div>

          <V2CheckpointList
            title="Interim checkpoints"
            checkpoints={interimCheckpoints}
            downloadURLBuilder={
              jobID
                ? (checkpointID) => v2API.checkpointDownloadURL(jobID, checkpointID)
                : undefined
            }
          />
        </>
      ) : null}

      {currentStep === 4 ? (
        <>
          <V2CheckpointList
            title="Final candidates"
            checkpoints={finalCandidates}
            selectedID={selectedCheckpointID}
            onSelect={setSelectedCheckpointID}
            onFeedback={(checkpointID, liked) => void handleCheckpointFeedback(checkpointID, liked)}
          />

          <div className="inline-actions">
            <button className="button button-secondary" type="button" onClick={() => setCurrentStep(3)}>
              Back to training
            </button>
            <button
              className="button"
              type="button"
              disabled={!selectedCheckpointID || !jobID || selectVersionMutation.isPending}
              onClick={() => void handleConfirmSelection()}
            >
              {selectVersionMutation.isPending ? "Confirming..." : "Confirm final version"}
            </button>
          </div>

          {selectionMessage ? <p className="muted">{selectionMessage}</p> : null}
        </>
      ) : null}
    </section>
  );
}
