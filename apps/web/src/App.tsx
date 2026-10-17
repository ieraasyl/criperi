import { useEffect, useState } from "react";
import { createNoteSchema, notesResponseSchema } from "@app/contracts";
import type { Note } from "@app/contracts";
import type { FormEvent } from "react";
import { authClient } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

function field(data: FormData, name: string) {
  const value = data.get(name);
  return typeof value === "string" ? value : "";
}

export default function App() {
  const { data: session, isPending } = authClient.useSession();
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notes, setNotes] = useState<Note[]>([]);
  const userId = session?.user.id;

  useEffect(() => {
    if (!userId) return;

    const controller = new AbortController();

    fetch("/api/notes", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not load notes");
        return notesResponseSchema.parse(await response.json());
      })
      .then((data) => setNotes(data.notes))
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : "Could not load notes");
        }
      });

    return () => controller.abort();
  }, [userId]);

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const fields = new FormData(event.currentTarget);
    const email = field(fields, "email");
    const password = field(fields, "password");

    setBusy(true);
    setError("");

    try {
      const result = register
        ? await authClient.signUp.email({
            name: field(fields, "name"),
            email,
            password,
          })
        : await authClient.signIn.email({ email, password });

      if (result.error) {
        throw new Error(result.error.message ?? "Authentication failed");
      }
      setNotes([]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Authentication failed");
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    setBusy(true);
    setError("");

    try {
      const result = await authClient.signOut();

      if (result.error) {
        throw new Error(result.error.message ?? "Could not sign out");
      }
      setNotes([]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not sign out");
    } finally {
      setBusy(false);
    }
  }

  async function submitNote(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const input = createNoteSchema.safeParse({
      title: new FormData(form).get("title"),
    });

    if (!input.success) {
      setError("Enter a title between 1 and 200 characters");
      return;
    }

    setBusy(true);
    setError("");

    try {
      const response = await fetch("/api/notes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input.data),
      });

      if (!response.ok) throw new Error("Could not save note");

      const list = await fetch("/api/notes");
      if (!list.ok) throw new Error("Could not reload notes");

      setNotes(notesResponseSchema.parse(await list.json()).notes);
      form.reset();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save note");
    } finally {
      setBusy(false);
    }
  }

  if (isPending) {
    return <main className="mx-auto max-w-lg p-6">Loading session…</main>;
  }

  return (
    <main className="mx-auto max-w-lg space-y-6 p-6">
      <h1 className="text-2xl font-semibold">Notes</h1>

      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}

      {session ? (
        <>
          <p>Signed in as {session.user.email}</p>
          <Button type="button" variant="outline" disabled={busy} onClick={signOut}>
            Sign out
          </Button>

          <form onSubmit={submitNote} className="space-y-3">
            <Label htmlFor="title">Note title</Label>
            <Input id="title" name="title" required maxLength={200} />
            <Button type="submit" disabled={busy}>
              Save note
            </Button>
          </form>

          <ul className="space-y-2">
            {notes
              .filter((note) => note.userId === userId)
              .map((note) => (
                <li key={note.id}>{note.title}</li>
              ))}
          </ul>
        </>
      ) : (
        <>
          <form onSubmit={submitAuth} className="space-y-4">
            {register && (
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input id="name" name="name" autoComplete="name" required />
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" name="email" type="email" autoComplete="email" required />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                name="password"
                type="password"
                autoComplete={register ? "new-password" : "current-password"}
                minLength={8}
                maxLength={128}
                required
              />
            </div>

            <Button type="submit" disabled={busy}>
              {register ? "Create account" : "Sign in"}
            </Button>
          </form>

          <Button
            type="button"
            variant="link"
            disabled={busy}
            onClick={() => {
              setRegister(!register);
              setError("");
            }}
          >
            {register ? "Sign in to an existing account" : "Create an account"}
          </Button>
        </>
      )}
    </main>
  );
}
