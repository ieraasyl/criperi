import { useContext } from "react";
import { createFileRoute, Outlet } from "@tanstack/react-router";
import { AuthenticatedQueryProvider } from "@/components/authenticated-query-provider";
import { SessionContext } from "@/lib/session-context";

export const Route = createFileRoute("/_authenticated")({
  beforeLoad: ({ context }) => {
    if (!context.session) {
      throw new Error("Не удалось подготовить сеанс. Обновите страницу.");
    }
    return { user: context.session.user };
  },
  component: AuthenticatedLayout,
});

function AuthenticatedLayout() {
  const session = useContext(SessionContext);
  const { user } = Route.useRouteContext();
  if (!session || session.user.id !== user.id) return null;
  return (
    <AuthenticatedQueryProvider key={user.id}>
      <Outlet />
    </AuthenticatedQueryProvider>
  );
}
