import { useState, type SubmitEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createNoteSchema,
  createNoteResponseSchema,
  notesResponseSchema,
  type CreateNote,
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
  const [titleError, setTitleError] = useState("");
  const queryClient = useQueryClient();
  const queryKey = ["notes", user.id];
  const notesQuery = useQuery({
    queryKey,
    queryFn: ({ signal }) => request("/api/notes", notesResponseSchema, { signal }),
  });
  const saveNote = useMutation({
    mutationFn: async (input: CreateNote) => {
      try {
        const { note } = await request("/api/notes", createNoteResponseSchema, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(input),
        });
        return note;
      } catch (cause) {
        if (
          cause instanceof ApiRequestError &&
          cause.code === "INVALID_RESPONSE" &&
          cause.status >= 200 &&
          cause.status < 300
        ) {
          // The write succeeded; reload instead of offering another POST.
          return undefined;
        }
        throw cause;
      }
    },
  });
  const signOut = useMutation({
    mutationFn: async () => {
      const result = await authClient.signOut();
      if (result.error) throw new Error(result.error.message ?? "Could not sign out");
    },
  });
  const busy = signOut.isPending || saveNote.isPending;
  const error = signOut.error?.message || saveNote.error?.message;
  const notes = notesQuery.data?.notes ?? [];
  const loading = notesQuery.isFetching && notesQuery.data === undefined;
  const loadError = !notesQuery.isFetching && notesQuery.error?.message;

  function submitNote(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    const form = event.currentTarget;
    signOut.reset();
    saveNote.reset();
    setTitleError("");
    const input = createNoteSchema.safeParse({
      title: new FormData(form).get("title"),
    });

    if (!input.success) {
      setTitleError("Enter a title between 1 and 200 characters");
      form.querySelector<HTMLInputElement>('input[name="title"]')?.focus();
      return;
    }

    saveNote.mutate(input.data, {
      onSuccess: async (note) => {
        form.reset();
        await queryClient.cancelQueries({ queryKey });
        if (note) {
          queryClient.setQueryData<{ notes: Note[] }>(queryKey, (current) => ({
            notes: [...(current?.notes ?? []).filter((item) => item.id !== note.id), note],
          }));
        }
        void queryClient.invalidateQueries({ queryKey });
      },
    });
  }

  return (
    <>
      {error && (
        <p role="alert" className="text-destructive">
          {error}
        </p>
      )}

      <p>Signed in as {user.email}</p>
      <Button
        type="button"
        variant="outline"
        disabled={busy}
        onClick={() => {
          saveNote.reset();
          signOut.mutate();
        }}
      >
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

      {saveNote.isSuccess && <p role="status">Note saved.</p>}
      {loading && <p role="status">Loading notes…</p>}
      {notesQuery.isFetching && !loading && <p role="status">Refreshing notes…</p>}
      {loadError && (
        <div>
          <p role="alert" className="text-destructive">
            Could not load notes. {loadError}
          </p>
          <Button type="button" variant="outline" onClick={() => void notesQuery.refetch()}>
            Retry loading notes
          </Button>
        </div>
      )}
      {!notesQuery.isPending && !loadError && notes.length === 0 && <p>No notes yet.</p>}

      <ul aria-label="Notes" aria-busy={notesQuery.isFetching} className="space-y-2">
        {notes
          .filter((note) => note.userId === user.id)
          .map((note) => (
            <li key={note.id}>{note.title}</li>
          ))}
      </ul>
    </>
  );
}
