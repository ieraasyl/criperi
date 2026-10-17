import { expect, test } from "vite-plus/test";
import { loginRedirect } from "./login-redirect";

test.each([
  ["/", "/"],
  ["/results?sort=recent#details", "/results?sort=recent#details"],
  ["/?filter=recent#results", "/?filter=recent#results"],
  ["/a/../results", "/results"],
])("preserves internal destination %s", (value, expected) => {
  expect(loginRedirect(value)).toBe(expected);
});

test.each([
  undefined,
  null,
  42,
  {},
  "",
  "results",
  "https://evil.example",
  "//evil.example",
  "/\\evil.example",
  "/results\\next",
  "/\n/evil.example",
  "/login",
  "/login/",
  "/LOGIN?redirect=/",
  "/%6cogin#top",
  "/a/../login",
  "/%5cevil.example",
  "/%zz",
])("rejects invalid or looping destination %s", (value) => {
  expect(loginRedirect(value)).toBe("/");
});
