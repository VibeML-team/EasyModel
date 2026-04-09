import { create } from "zustand";

type SessionState = {
  sessionID: string | null;
  setSessionID: (sessionID: string | null) => void;
};

export const useSessionStore = create<SessionState>((set) => ({
  sessionID: null,
  setSessionID: (sessionID) => set({ sessionID }),
}));
