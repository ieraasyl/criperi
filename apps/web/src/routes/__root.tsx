import {
  createRootRouteWithContext,
  Link,
  Outlet,
  useRouter,
  type ErrorComponentProps,
} from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import type { Session } from "@/lib/session-context";

export const Route = createRootRouteWithContext<{ session: Session | null }>()({
  component: RootLayout,
  notFoundComponent: NotFound,
  errorComponent: RouteError,
});

function RootLayout() {
  return (
    <main className="mx-auto max-w-4xl space-y-6 p-6">
      <Outlet />
    </main>
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
