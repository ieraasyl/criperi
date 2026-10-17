import { createFileRoute } from "@tanstack/react-router";
import { Notes } from "@/components/notes";

export const Route = createFileRoute("/_authenticated/")({ component: Index });

function Index() {
  const { user } = Route.useRouteContext();
  return (
    <>
      <h1 className="text-2xl font-semibold">Notes</h1>
      <Notes user={user} />
    </>
  );
}
