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
    <main className="mx-auto max-w-lg space-y-6 p-6">
      <Outlet />
    </main>
  );
}

function NotFound() {
  return (
    <>
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <Link to="/">Go home</Link>
    </>
  );
}

function RouteError({ reset }: ErrorComponentProps) {
  const router = useRouter();
  return (
    <>
      <p role="alert">Could not load this page.</p>
      <Button
        onClick={async () => {
          await router.invalidate();
          reset();
        }}
      >
        Retry
      </Button>
      <Link to="/">Go home</Link>
    </>
  );
}
