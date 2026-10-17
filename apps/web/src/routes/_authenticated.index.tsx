import { createFileRoute } from "@tanstack/react-router";
import { UploadAnalysis } from "@/components/analysis";

export const Route = createFileRoute("/_authenticated/")({ component: Index });

function Index() {
  const { user } = Route.useRouteContext();
  return <UploadAnalysis email={user.email} />;
}
