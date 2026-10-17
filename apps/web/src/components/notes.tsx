import { useEffect, useState, type SubmitEvent } from "react";
import { createNoteSchema, notesResponseSchema, type Note } from "@app/contracts";
import { authClient } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";

type NotesProps = {
  user: { id: string; email: string };
};

export function Notes({ user }: NotesProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [titleError, setTitleError] = useState("");
  const [notes, setNotes] = useState<Note[]>([]);
  const userId = user.id;

  useEffect(() => {
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

  async function signOut() {
    setBusy(true);
    setError("");

    try {
      const result = await authClient.signOut();

      if (result.error) {
        throw new Error(result.error.message ?? "Could not sign out");
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not sign out");
    } finally {
      setBusy(false);
    }
  }

  async function submitNote(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    setError("");
    setTitleError("");
    const input = createNoteSchema.safeParse({
      title: new FormData(form).get("title"),
    });

    if (!input.success) {
      setTitleError("Enter a title between 1 and 200 characters");
      form.querySelector<HTMLInputElement>('input[name="title"]')?.focus();
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

  return (
    <>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}

      <p>Signed in as {user.email}</p>
      <Button type="button" variant="outline" disabled={busy} onClick={signOut}>
        Sign out
      </Button>

      <form noValidate onSubmit={submitNote}>
        <FieldGroup>
          <Field data-invalid={!!titleError}>
            <FieldLabel htmlFor="title">Note title</FieldLabel>
            <Input
              id="title"
              name="title"
              required
              maxLength={200}
              aria-invalid={!!titleError}
              aria-describedby={titleError ? "title-error" : undefined}
              onChange={() => setTitleError("")}
            />
            <FieldError id="title-error">{titleError}</FieldError>
          </Field>
          <Field>
            <Button type="submit" disabled={busy}>
              Save note
            </Button>
          </Field>
        </FieldGroup>
      </form>

      <ul className="space-y-2">
        {notes
          .filter((note) => note.userId === userId)
          .map((note) => (
            <li key={note.id}>{note.title}</li>
          ))}
      </ul>
    </>
  );
}
