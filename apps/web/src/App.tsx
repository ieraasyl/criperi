import { AuthForm } from "@/components/auth-form";
import { Notes } from "@/components/notes";
import { authClient } from "@/lib/auth-client";

export default function App() {
  const { data: session, isPending } = authClient.useSession();

  if (isPending) {
    return <main className="mx-auto max-w-lg p-6">Loading session…</main>;
  }

  return (
    <main className="mx-auto max-w-lg space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Notes</h1>
      {session ? <Notes key={session.user.id} user={session.user} /> : <AuthForm />}
    </main>
  );
}
