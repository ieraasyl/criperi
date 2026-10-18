import { createFileRoute } from "@tanstack/react-router";
import { AnalysisResult } from "@/components/analysis";

export const Route = createFileRoute("/_authenticated/analyses/$jobId")({
  validateSearch: (search: Record<string, unknown>): { event?: number; frame?: number } => {
    const position = (value: unknown) => {
      if (typeof value !== "string" && typeof value !== "number") return undefined;
      const parsed = Number(value);
      return Number.isSafeInteger(parsed) && parsed > 0 && parsed <= 100_000 ? parsed : undefined;
    };
    return { event: position(search.event), frame: position(search.frame) };
  },
  component: Result,
});

function Result() {
  const { jobId } = Route.useParams();
  const { user } = Route.useRouteContext();
  return <AnalysisResult userId={user.id} jobId={jobId} />;
}
