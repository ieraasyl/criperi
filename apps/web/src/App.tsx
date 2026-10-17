import { useEffect, useState } from "react";
import { RouterProvider } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { authClient } from "@/lib/auth-client";
import { SessionContext } from "@/lib/session-context";
import { createAppRouter } from "@/router";

export default function App({
  router: suppliedRouter,
}: {
  router?: ReturnType<typeof createAppRouter>;
}) {
  const [router] = useState(() => suppliedRouter ?? createAppRouter());
  const { data: session, isPending, error, refetch } = authClient.useSession();
  const userId = session?.user.id;

  useEffect(() => {
    if (!isPending && !error) void router.invalidate();
  }, [router, userId, isPending, error]);

  if (isPending) {
    return (
      <main className="mx-auto max-w-lg p-6" role="status">
        Проверка входа…
      </main>
    );
  }

  if (error) {
    return (
      <main className="mx-auto max-w-lg space-y-6 p-6">
        <p role="alert">Не удалось проверить вход.</p>
        <Button onClick={() => void refetch()}>Повторить проверку</Button>
      </main>
    );
  }

  return (
    <SessionContext value={session}>
      <RouterProvider router={router} context={{ session }} />
    </SessionContext>
  );
}
