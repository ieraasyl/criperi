import type { ReviewJob } from "@app/contracts";

export function analysisPollingInterval(job: ReviewJob | undefined) {
  if (!job) return 2000;
  return job.status === "processing" && Date.parse(job.expires_at) > Date.now() ? 2000 : false;
}
