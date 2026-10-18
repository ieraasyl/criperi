import { useEffect, useRef, useState } from "react";
import { RouterProvider } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { authClient } from "@/lib/auth-client";
import { SessionContext } from "@/lib/session-context";
import { createAppRouter } from "@/router";
import { ensureVisitorSession } from "@/lib/visitor-session";

export default function App({
  router: suppliedRouter,
}: {
  router?: ReturnType<typeof createAppRouter>;
}) {
  const [router] = useState(() => suppliedRouter ?? createAppRouter());
  const { data: session, isPending, error, refetch } = authClient.useSession();
  const userId = session?.user.id;
  const [visitorError, setVisitorError] = useState("");
  const started = useRef(false);
  const mounted = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (userId) {
      started.current = false;
      return;
    }
    if (isPending || error || started.current) return;
    // Keep this attempt across the auth client's pending/refetch transitions.
    started.current = true;
    void ensureVisitorSession()
      .then(() => {
        if (mounted.current) {
          setVisitorError("");
          return refetch();
        }
      })
      .catch((cause: unknown) => {
        if (mounted.current)
          setVisitorError(cause instanceof Error ? cause.message : "Не удалось подготовить сеанс.");
      });
  }, [userId, isPending, error, refetch, visitorError]);

  useEffect(() => {
    if (!isPending && !error && userId) void router.invalidate();
  }, [router, userId, isPending, error]);

  const failure = (!session && visitorError) || (error ? "Не удалось проверить сеанс." : "");
  if (failure) {
    return (
      <main className="mx-auto max-w-lg space-y-4 p-4 sm:p-6">
        <p role="alert">{failure}</p>
        <Button
          onClick={() => {
            if (!session && visitorError) {
              started.current = false;
              setVisitorError("");
              if (error) void refetch();
            } else void refetch();
          }}
        >
          Повторить
        </Button>
      </main>
    );
  }

  if (isPending || !session) {
    return (
      <main className="mx-auto max-w-lg p-4 sm:p-6" role="status">
        Подготовка сеанса…
      </main>
    );
  }

  return (
    <SessionContext value={session}>
      <RouterProvider router={router} context={{ session }} />
    </SessionContext>
  );
}
