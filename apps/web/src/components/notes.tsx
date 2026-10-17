import { useCallback, useEffect, useRef, useState, type SubmitEvent } from "react";
import {
  createNoteSchema,
  createNoteResponseSchema,
  notesResponseSchema,
  type Note,
} from "@app/contracts";
import { authClient } from "@/lib/auth-client";
import { ApiRequestError, request } from "@/lib/request";
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
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [saved, setSaved] = useState(false);
  const loadController = useRef<AbortController | null>(null);
  const active = useRef(false);
  const userId = user.id;

  const loadNotes = useCallback(() => {
    loadController.current?.abort();
    const controller = new AbortController();
    loadController.current = controller;

    return request("/api/notes", notesResponseSchema, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setNotes(data.notes);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setLoadError(cause instanceof Error ? cause.message : "Could not load notes");
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
  }, []);

  function refreshNotes() {
    setLoading(true);
    setLoadError("");
    void loadNotes();
  }

  useEffect(() => {
    active.current = true;
    void loadNotes();
    return () => {
      active.current = false;
      loadController.current?.abort();
    };
  }, [userId, loadNotes]);

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
    setSaved(false);
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
      const { note } = await request("/api/notes", createNoteResponseSchema, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input.data),
      });

      if (!active.current) return;
      loadController.current?.abort();
      setNotes((current) => [...current.filter((item) => item.id !== note.id), note]);
      form.reset();
      setSaved(true);
      refreshNotes();
    } catch (cause) {
      if (!active.current) return;
      if (cause instanceof ApiRequestError && cause.status >= 200 && cause.status < 300) {
        // The write succeeded even if its response could not be parsed. Do not offer a duplicate save.
        loadController.current?.abort();
        form.reset();
        setSaved(true);
        refreshNotes();
      } else {
        setError(cause instanceof Error ? cause.message : "Could not save note");
      }
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
              disabled={busy}
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

      {saved && <p role="status">Note saved.</p>}
      {loading && <p role="status">Loading notes…</p>}
      {loadError && (
        <div>
          <p role="alert" className="text-destructive">
            Could not load notes. {loadError}
          </p>
          <Button type="button" variant="outline" onClick={refreshNotes}>
            Retry loading notes
          </Button>
        </div>
      )}
      {!loading && !loadError && notes.length === 0 && <p>No notes yet.</p>}

      <ul aria-label="Notes" aria-busy={loading} className="space-y-2">
        {notes
          .filter((note) => note.userId === userId)
          .map((note) => (
            <li key={note.id}>{note.title}</li>
          ))}
      </ul>
    </>
  );
}
