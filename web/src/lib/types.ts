export interface Submission {
  id: number;
  assessment_id: number;
  student_id: string;
  started_at: string;
}

export interface AnswerOut {
  id: number;
  question_id: number;
  frq_part_id?: number;
  answer_text: string;
  graded: boolean;
}

export interface Report {
  average: number;
  min: number;
  max: number;
  median: number;
  stddev: number;
  question_stats: { question_id: number; avg_score: number }[];
}

export interface Class {
  id: string;
  name: string;
  join_code: string;
}