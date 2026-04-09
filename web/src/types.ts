export type JsonPrimitive = string | number | boolean | null;
export type Json = JsonPrimitive | Json[] | { [key: string]: Json };

export type APIHealth = {
  status: string;
  version?: string;
  deploy_tag?: string;
  llm_configured?: boolean;
  llm_model?: string | null;
};

export type V2Domain = "general" | "ai4science" | "gnn" | "timeseries" | "rl";

export type V2ClarificationQuestion = {
  id: string;
  text: string;
  type: "open" | "multiple_choice" | "confirm" | string;
  options?: string[];
  context?: string;
  priority?: number;
};

export type V2IntentCard = {
  task: {
    description: string;
    type: string;
    confidence: number;
  };
  constraints: {
    must_keep: string[];
    can_change: string[];
    priority: string;
    error_preference: string;
  };
  data: {
    summary: Record<string, unknown>;
    augmentation_style: string;
  };
  acceptance: {
    success_criteria: string[];
    unacceptable_errors: string[];
  };
  system_notes: {
    reasoning: string;
    uncertainties: string[];
  };
  clarification: {
    is_ambiguous: boolean;
    ambiguity_score: number;
    missing_elements: string[];
    unclear_aspects: string[];
    suggested_questions: V2ClarificationQuestion[];
    can_proceed: boolean;
  };
};

export type V2DataCard = {
  overview: {
    total: number;
    train: number;
    val: number;
    issues: Array<Record<string, unknown>>;
  };
  sample_control: {
    high_priority_count: number;
    excluded_count: number;
    can_review: boolean;
  };
  variations: {
    allowed: string[];
    forbidden: string[];
    style: string;
  };
  pairings: {
    count: number;
    status: string;
  };
  previews: Array<Record<string, unknown>>;
};

export type V2PlanCard = {
  approach: {
    name: string;
    reasoning: string;
  };
  goals: {
    primary: string;
    secondary: string[];
  };
  preferences: {
    quality_speed_cost: string;
    deployment: string;
  };
};

export type V2CheckpointCandidate = {
  id: string;
  step: number;
  metric: number;
  metric_name?: string;
  model?: string;
  status?: string;
};

export type V2CompileResponse = {
  session_id: string;
  summary: string;
  can_proceed: boolean;
  intent_card: V2IntentCard;
  data_card: V2DataCard;
  plan_card: V2PlanCard;
};

export type V2ClarifyResponse = {
  session_id: string;
  clarification_complete: boolean;
  message: string;
  intent_card: V2IntentCard;
  remaining_questions: V2ClarificationQuestion[];
};

export type V2StartTrainingResponse = {
  job_id: string;
  status: string;
  message: string;
};

export type V2CheckpointListResponse = {
  job_id: string;
  checkpoints: V2CheckpointCandidate[];
  count: number;
};

export type V2PauseAndDownloadResponse = {
  success: boolean;
  message: string;
  checkpoints: V2CheckpointCandidate[];
  best_checkpoint?: V2CheckpointCandidate;
  download_url?: string;
};

export type V2FinalizeResponse = {
  job_id: string;
  acceptance_card: {
    candidates: V2CheckpointCandidate[];
    comparison: string[];
    selected: string;
    feedback_summary?: {
      total_feedback: number;
      positive: number;
    };
    final?: {
      path: string;
      deployment: Record<string, unknown>;
    } | null;
  };
  message: string;
};

export type V2SelectVersionResponse = {
  job_id: string;
  selected_checkpoint: string;
  model_path: string;
  deployment_recommendations: Record<string, unknown>;
};

export type V2TrainingStatus = {
  job_id: string;
  status: string;
  progress_percent: number;
  observation: {
    progress?: {
      stage?: string;
      current?: number;
      total?: number;
      eta?: number;
      percent?: number;
    };
    metrics?: Record<string, unknown>;
    best_checkpoint?: {
      step?: number;
      metric?: number;
    };
    alerts?: Array<{
      type: string;
      message: string;
    }>;
  } | null;
};
