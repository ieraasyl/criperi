import { z } from "zod";

export const createNoteSchema = z.object({
  title: z.string().trim().min(1).max(200),
});

export const aiPromptSchema = z.object({
  prompt: z.string().trim().min(1).max(2000),
});

export type CreateNote = z.infer<typeof createNoteSchema>;

export const apiErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    issues: z.array(z.object({ path: z.array(z.string()), message: z.string() })).optional(),
  }),
});

export type ApiError = z.infer<typeof apiErrorSchema>;
