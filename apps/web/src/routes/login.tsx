import { createFileRoute, redirect } from "@tanstack/react-router";
import { AuthForm } from "@/components/auth-form";
import { loginRedirect } from "@/lib/login-redirect";

export const Route = createFileRoute("/login")({
  validateSearch: (search: Record<string, unknown>): { redirect?: string } => ({
    redirect: typeof search.redirect === "string" ? loginRedirect(search.redirect) : undefined,
  }),
  beforeLoad: ({ context, search }) => {
    if (context.session) throw redirect({ href: loginRedirect(search.redirect), replace: true });
  },
  component: Login,
});

function Login() {
  return <AuthForm />;
}
