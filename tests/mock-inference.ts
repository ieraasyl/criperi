import { createServer } from "node:http";
import { randomBytes, randomUUID } from "node:crypto";
import evidence from "./fixtures/evidence.json";

export async function startMockInference() {
  const key = randomBytes(32).toString("hex");
  const jobs = new Map<string, { name: string }>();
  const server = createServer(async (req, res) => {
    if (req.headers["x-api-key"] !== key) {
      res.writeHead(401).end();
      return;
    }
    if (req.url?.startsWith("/__test/submissions/")) {
      const name = req.url.split("/").at(-1);
      const ids = [...jobs].filter(([, job]) => job.name === name).map(([id]) => id);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ submissions: ids.length, ids }));
      return;
    }
    if (req.method === "POST" && req.url === "/api/v1/live") {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const { frames } = JSON.parse(Buffer.concat(chunks).toString());
      const first = frames[0].timestamp_sec;
      const last = frames.at(-1).timestamp_sec;
      res.writeHead(200, { "Content-Type": "text/event-stream" });
      res.write('data: {"type":"stage","stage":"detecting"}\n\n');
      if (first === 1000) {
        res.end();
        return;
      }
      if (first === 1100) {
        res.end(
          'data: {"type":"error","error":{"code":"GPU_ERROR","message":"private path /model/token-secret"}}\n\n',
        );
        return;
      }
      if (first === 1200) {
        res.end("data: not JSON\n\n");
        return;
      }
      res.write(
        `data: ${JSON.stringify({ type: "frame", ...frames.at(-1), people_count: 1, objects_count: 2 })}\n\n`,
      );
      res.end(
        `data: ${JSON.stringify({
          type: "result",
          from_sec: first,
          to_sec: first === 1300 ? last + 100 : last,
          violations: [
            {
              type: "V17. Еда или напиток",
              timecode_sec: last,
              description: "На рабочем столе виден напиток.",
              evidence_frames: [frames.at(-1)],
            },
          ],
        })}\n\n`,
      );
      return;
    }
    if (req.method === "POST" && req.url === "/api/v1/analyze") {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      const request = new Request("http://mock/upload", {
        method: "POST",
        headers: { "Content-Type": String(req.headers["content-type"]) },
        body: Buffer.concat(chunks),
      });
      const form = await request.formData();
      const file = form.get("file");
      if (!(file instanceof File) || !file.name.endsWith(".mov") || file.name === "invalid.mov") {
        res.writeHead(422).end();
        return;
      }
      if (file.name === "busy.mov") {
        res.writeHead(429).end();
        return;
      }
      const id = `fc-${randomUUID().replaceAll("-", "")}`;
      jobs.set(id, { name: file.name });
      if (file.name === "lost-ack.mov") {
        req.socket.destroy();
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        file.name === "malformed-ack.mov"
          ? "invalid JSON"
          : JSON.stringify({ job_id: id, status: "processing" }),
      );
      return;
    }
    const jobId = req.url?.split("/").at(-1) ?? "";
    const job = jobs.get(jobId);
    if (!job) {
      res.writeHead(404).end();
      return;
    }
    if (job.name === "outage.mov") {
      res.writeHead(503).end();
      return;
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    if (job.name === "processing.mov") {
      res.end(JSON.stringify({ job_id: jobId, status: "processing" }));
      return;
    }
    if (job.name === "failed.mov") {
      res.end(
        JSON.stringify({
          job_id: jobId,
          status: "failed",
          error: "RuntimeError: GPU unavailable at /private/container/path",
        }),
      );
      return;
    }
    const event = {
      type: "Личные вещи на рабочем месте",
      timecode_sec: 1,
      description:
        "На рабочем столе видна личная вещь. Источник: НБРК №231, п. 12. Требуется проверка.",
      evidence_frames: [{ timestamp_sec: 1, image_base64: evidence.jpeg }],
    };
    res.end(
      JSON.stringify({
        job_id: jobId,
        status: "done",
        verdict: job.name === "negative.mov" ? "no_violation" : "violation",
        violations:
          job.name === "negative.mov"
            ? []
            : job.name === "bad-result.mov"
              ? [{ ...event, timecode_sec: -1 }]
              : [event, { ...event, timecode_sec: 3 }],
      }),
    );
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Mock did not start");
  return {
    key,
    origin: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve())),
      ),
  };
}
