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

test("RPC exposes analysis endpoints and validated review inputs", () => {
  expectTypeOf(api.health.$get).toBeFunction();
  expectTypeOf(api.analyses.$post).toBeFunction();
  expectTypeOf(api.v1.analyze.$post).toBeFunction();
  const save = api.analyses[":job_id"].reviews[":event_index"].$put;
  expectTypeOf<InferRequestType<typeof save>["json"]>().toEqualTypeOf<{
    status: "confirmed" | "rejected" | "needs_review";
    comment?: string | undefined;
  }>();

  // @ts-expect-error Auth uses Better Auth's client.
  void api.auth;
  // @ts-expect-error Unknown endpoints must not appear on the client.
  void api.missing;
});

test("RPC infers successful review responses and serialized timestamps", () => {
  const save = api.analyses[":job_id"].reviews[":event_index"].$put;
  type Saved = InferResponseType<typeof save, 200>;
  const update = () =>
    request((init) =>
      save(
        {
          param: { job_id: "280312ee-e063-42b9-a846-4a18fe361682", event_index: "0" },
          json: { status: "confirmed", comment: "Проверено." },
        },
        { init },
      ),
    );

  expectTypeOf<Saved["review"]["updated_at"]>().toEqualTypeOf<string>();
  expectTypeOf<Awaited<ReturnType<typeof update>>>().toEqualTypeOf<Saved>();
  expectTypeOf<InferResponseType<typeof api.health.$get, 200>>().toEqualTypeOf<{ ok: true }>();
});
