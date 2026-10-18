import { useState, type SubmitEvent } from "react";
import { authClient } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";

function field(data: FormData, name: string) {
  const value = data.get(name);
  return typeof value === "string" ? value : "";
}

export function AuthForm() {
  const [register, setRegister] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string | undefined>>({});

  async function submitAuth(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    const form = event.currentTarget;
    const invalidInputs = Array.from(form.querySelectorAll("input")).filter(
      (input) => !input.validity.valid,
    );

    setError("");
    setFieldErrors(
      Object.fromEntries(
        invalidInputs.map((input) => [
          input.name,
          input.validity.valueMissing
            ? "Заполните это поле."
            : input.validity.typeMismatch
              ? "Введите адрес почты, например name@example.com."
              : input.validity.tooShort || input.validity.tooLong
                ? "Пароль должен содержать от 8 до 128 символов."
                : "Проверьте значение поля.",
        ]),
      ),
    );

    if (invalidInputs.length) {
      invalidInputs[0]?.focus();
      return;
    }

    const fields = new FormData(form);
    const email = field(fields, "email");
    const password = field(fields, "password");

    setBusy(true);

    try {
      const result = register
        ? await authClient.signUp.email({
            name: field(fields, "name"),
            email,
            password,
          })
        : await authClient.signIn.email({ email, password });

      if (result.error) {
        setError(
          register
            ? "Не удалось создать аккаунт. Проверьте введённые данные."
            : "Не удалось войти. Проверьте данные аккаунта.",
        );
      }
    } catch {
      setError("Не удалось подключиться. Проверьте соединение и повторите попытку.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="mx-auto max-w-md space-y-5 py-4">
      <h1 className="text-2xl font-semibold">{register ? "Создать аккаунт" : "Вход в аккаунт"}</h1>
      <form
        aria-busy={busy}
        noValidate
        onSubmit={submitAuth}
        onInput={(event) => {
          const input = event.target;
          if (input instanceof HTMLInputElement) {
            setFieldErrors((errors) => ({ ...errors, [input.name]: undefined }));
          }
        }}
      >
        <FieldGroup>
          {register && (
            <Field data-invalid={!!fieldErrors.name}>
              <FieldLabel htmlFor="name">Имя</FieldLabel>
              <Input
                id="name"
                name="name"
                autoComplete="name"
                disabled={busy}
                className="h-11"
                required
                aria-invalid={!!fieldErrors.name}
                aria-describedby={fieldErrors.name ? "name-error" : undefined}
              />
              <FieldError id="name-error">{fieldErrors.name}</FieldError>
            </Field>
          )}

          <Field data-invalid={!!fieldErrors.email}>
            <FieldLabel htmlFor="email">Электронная почта</FieldLabel>
            <Input
              id="email"
              name="email"
              type="email"
              autoComplete="email"
              placeholder="name@example.com"
              disabled={busy}
              className="h-11"
              required
              aria-invalid={!!fieldErrors.email}
              aria-describedby={fieldErrors.email ? "email-error" : undefined}
            />
            <FieldError id="email-error">{fieldErrors.email}</FieldError>
          </Field>

          <Field data-invalid={!!fieldErrors.password}>
            <FieldLabel htmlFor="password">Пароль</FieldLabel>
            <Input
              id="password"
              name="password"
              type="password"
              autoComplete={register ? "new-password" : "current-password"}
              minLength={8}
              maxLength={128}
              required
              disabled={busy}
              className="h-11"
              aria-invalid={!!fieldErrors.password}
              aria-describedby={
                fieldErrors.password ? "password-error" : register ? "password-help" : undefined
              }
            />
            {register && (
              <FieldDescription id="password-help">От 8 до 128 символов.</FieldDescription>
            )}
            <FieldError id="password-error">{fieldErrors.password}</FieldError>
          </Field>

          <Field>
            <FieldError>{error}</FieldError>
            <Button type="submit" disabled={busy} className="h-11">
              {busy
                ? register
                  ? "Создаём аккаунт…"
                  : "Входим…"
                : register
                  ? "Зарегистрироваться"
                  : "Войти"}
            </Button>
          </Field>
        </FieldGroup>
      </form>

      <div className="border-t pt-4 text-center">
        <p className="text-sm text-muted-foreground">
          {register ? "Уже есть аккаунт?" : "Нет аккаунта?"}
        </p>
        <Button
          type="button"
          variant="link"
          disabled={busy}
          onClick={() => {
            setRegister(!register);
            setError("");
            setFieldErrors({});
          }}
        >
          {register ? "Войти в существующий аккаунт" : "Создать аккаунт"}
        </Button>
      </div>
    </section>
  );
}
