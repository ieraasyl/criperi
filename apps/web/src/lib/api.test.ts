import { expectTypeOf, test } from "vite-plus/test";
import type { InferRequestType, InferResponseType } from "hono/client";
import { api } from "./api";
import { request } from "./request";

test("RPC source types preserve browser globals", () => {
  function browserTypes() {
    document.body.append(document.createElement("div"));
    // @ts-expect-error HTMLRewriter methods are not browser DOM methods.
    document.body.setInnerContent("hello");
  }

  expectTypeOf(browserTypes).returns.toBeVoid();
  // @ts-expect-error Worker database bindings must not become browser globals.
  expectTypeOf<D1Database>().toBeAny();
  // @ts-expect-error Worker environment bindings belong to the API project.
  expectTypeOf<CloudflareBindings>().toBeAny();
});

test("RPC exposes the application endpoints and validated request inputs", () => {
  expectTypeOf(api.health.$get).toBeFunction();
  expectTypeOf(api.me.$get).toBeFunction();
  expectTypeOf(api.notes.$get).toBeFunction();
  expectTypeOf(api.analyses.$post).toBeFunction();
  expectTypeOf(api.v1.analyze.$post).toBeFunction();
  expectTypeOf<InferRequestType<typeof api.notes.$post>["json"]>().toEqualTypeOf<{
    title: string;
  }>();

  // @ts-expect-error The auth handler uses Better Auth's client.
  void api.auth;
  // @ts-expect-error Unknown endpoints must not appear on the client.
  void api.missing;
  // @ts-expect-error Validated titles must be strings.
  const invalid: InferRequestType<typeof api.notes.$post> = { json: { title: 123 } };
  void invalid;
});

test("RPC infers successful JSON bodies and serialized dates", () => {
  type Notes = InferResponseType<typeof api.notes.$get, 200>;
  type Created = InferResponseType<typeof api.notes.$post, 201>;
  const create = () => request((init) => api.notes.$post({ json: { title: "Example" } }, { init }));

  expectTypeOf<Notes["notes"][number]["createdAt"]>().toEqualTypeOf<string>();
  expectTypeOf<Created["note"]>().toExtend<Notes["notes"][number]>();
  expectTypeOf<Awaited<ReturnType<typeof create>>>().toEqualTypeOf<Created>();
  expectTypeOf<InferResponseType<typeof api.health.$get, 200>>().toEqualTypeOf<{ ok: true }>();
});
