import type { APIHealth } from "../types";
import { requestJSON } from "../lib/http";

export const v1API = {
  health: () => requestJSON<APIHealth>("/health"),
};
