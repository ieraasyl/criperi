import {
  createRootRouteWithContext,
  Link,
  Outlet,
  useRouter,
  useRouterState,
  type ErrorComponentProps,
} from "@tanstack/react-router";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import type { Session } from "@/lib/session-context";

export const Route = createRootRouteWithContext<{ session: Session | null }>()({
  component: RootLayout,
  notFoundComponent: NotFound,
  errorComponent: RouteError,
});

function RootLayout() {
  const reviewing = useRouterState({
    select: (state) => state.location.pathname.startsWith("/analyses/"),
  });
  return (
    <TooltipProvider>
      <div
        className={`mx-auto w-full p-4 sm:px-6 ${reviewing ? "max-w-7xl lg:px-10" : "max-w-6xl"}`}
      >
        <main className="space-y-4">
          <Outlet />
        </main>
      </div>
    </TooltipProvider>
  );
}

function NotFound() {
  return (
    <>
      <h1 className="text-2xl font-semibold">Страница не найдена</h1>
      <Link to="/">На главную</Link>
    </>
  );
}

function RouteError({ reset }: ErrorComponentProps) {
  const router = useRouter();
  return (
    <>
      <p role="alert">Не удалось загрузить страницу.</p>
      <Button
        onClick={async () => {
          await router.invalidate();
          reset();
        }}
      >
        Повторить
      </Button>
      <Link to="/">На главную</Link>
    </>
  );
}
