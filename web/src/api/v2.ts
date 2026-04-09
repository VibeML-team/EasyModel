import { requestJSON } from "../lib/http";
import type {
  V2CheckpointListResponse,
  V2ClarifyResponse,
  V2CompileResponse,
  V2Domain,
  V2FinalizeResponse,
  V2PauseAndDownloadResponse,
  V2SelectVersionResponse,
  V2StartTrainingResponse,
  V2TrainingStatus,
} from "../types";

export type CompileInput = {
  intent: string;
  domain: V2Domain;
  data_preview?: Record<string, unknown>;
};

export type ClarifyInput = {
  session_id: string;
  answers: Array<{
    question_id: string;
    answer: string;
  }>;
};

export const v2API = {
  compile: (payload: CompileInput) =>
    requestJSON<V2CompileResponse>("/v2/compile", {
      method: "POST",
      body: payload,
    }),
  clarify: (payload: ClarifyInput) =>
    requestJSON<V2ClarifyResponse>("/v2/clarify", {
      method: "POST",
      body: payload,
    }),
  startTraining: (payload: { session_id: string; dataset_id?: string }) =>
    requestJSON<V2StartTrainingResponse>("/v2/train/start", {
      method: "POST",
      body: payload,
    }),
  trainingStatus: (jobID: string) => requestJSON<V2TrainingStatus>(`/v2/train/status/${jobID}`),
  pauseTraining: (jobID: string) =>
    requestJSON<{ success: boolean; message: string }>(`/v2/train/pause/${jobID}`, {
      method: "POST",
    }),
  resumeTraining: (jobID: string) =>
    requestJSON<{ success: boolean; message: string }>(`/v2/train/resume/${jobID}`, {
      method: "POST",
    }),
  listCheckpoints: (jobID: string) => requestJSON<V2CheckpointListResponse>(`/v2/train/checkpoints/${jobID}`),
  pauseAndDownload: (jobID: string) =>
    requestJSON<V2PauseAndDownloadResponse>(`/v2/train/pause-and-download/${jobID}`, {
      method: "POST",
    }),
  finalizeTraining: (jobID: string) =>
    requestJSON<V2FinalizeResponse>(`/v2/train/finalize/${jobID}`, {
      method: "POST",
    }),
  feedback: (payload: { job_id: string; checkpoint_id: string; liked: boolean; reason?: string }) =>
    requestJSON<{ success: boolean; message: string }>("/v2/train/feedback", {
      method: "POST",
      body: payload,
    }),
  selectVersion: (payload: { job_id: string; checkpoint_id: string }) =>
    requestJSON<V2SelectVersionResponse>("/v2/train/select", {
      method: "POST",
      body: payload,
    }),
  checkpointDownloadURL: (jobID: string, checkpointID: string) =>
    `/api/v2/train/checkpoints/${jobID}/${checkpointID}/download`,
};
