import { create } from "zustand";
import { v2API } from "../api/v2";

type TrainingState = {
  jobID: string | null;
  status: string;
  progressPercent: number;
  observation: Record<string, unknown> | null;
  loading: boolean;
  error: string | null;
  pollHandle: number | null;
  setJobID: (jobID: string | null) => void;
  stopPolling: () => void;
  pollStatus: () => Promise<void>;
  startPolling: (intervalMs?: number) => Promise<void>;
};

export const useTrainingStore = create<TrainingState>((set, get) => ({
  jobID: null,
  status: "idle",
  progressPercent: 0,
  observation: null,
  loading: false,
  error: null,
  pollHandle: null,
  setJobID: (jobID) => {
    if (!jobID) {
      const handle = get().pollHandle;
      if (handle) {
        window.clearInterval(handle);
      }
      set({
        jobID: null,
        status: "idle",
        progressPercent: 0,
        observation: null,
        pollHandle: null,
      });
      return;
    }
    set({ jobID });
  },
  stopPolling: () => {
    const handle = get().pollHandle;
    if (handle) {
      window.clearInterval(handle);
    }
    set({ pollHandle: null });
  },
  pollStatus: async () => {
    const jobID = get().jobID;
    if (!jobID) {
      return;
    }

    set({ loading: true, error: null });
    try {
      const status = await v2API.trainingStatus(jobID);
      set({
        status: status.status,
        progressPercent: status.progress_percent,
        observation: status.observation,
        loading: false,
      });
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : "Failed to poll status",
      });
    }
  },
  startPolling: async (intervalMs = 2_000) => {
    get().stopPolling();
    await get().pollStatus();
    const handle = window.setInterval(() => {
      void get().pollStatus();
    }, intervalMs);
    set({ pollHandle: handle });
  },
}));
