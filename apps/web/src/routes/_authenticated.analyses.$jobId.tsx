import { createFileRoute } from "@tanstack/react-router";
import { AnalysisResult } from "@/components/analysis";

export const Route = createFileRoute("/_authenticated/analyses/$jobId")({ component: Result });

function Result() {
  const { jobId } = Route.useParams();
  const { user } = Route.useRouteContext();
  return <AnalysisResult userId={user.id} jobId={jobId} />;
}
