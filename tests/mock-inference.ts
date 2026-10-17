import { createServer } from "node:http";
import { randomBytes } from "node:crypto";
import evidence from "./fixtures/evidence.json";

export async function startMockInference() {
  const key = randomBytes(32).toString("hex");
  const jobs = new Map<string, { name: string; submissions: number }>();
  const server = createServer(async (req, res) => {
    if (req.headers["x-api-key"] !== key) {
      res.writeHead(401).end();
      return;
    }
    const id = req.headers["x-job-id"] as string;
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
      const existing = jobs.get(id);
      jobs.set(id, { name: file.name, submissions: (existing?.submissions ?? 0) + 1 });
      if (file.name === "lost-ack.mov") {
        req.socket.destroy();
        return;
      }
      res.writeHead(202, { "Content-Type": "application/json" });
      res.end(file.name === "malformed-ack.mov" ? "invalid JSON" : JSON.stringify({ job_id: id }));
      return;
    }
    const jobId = req.url?.split("/").at(-1) ?? "";
    const job = jobs.get(jobId);
    if (!job) {
      res.writeHead(404).end();
      return;
    }
    if (req.url?.startsWith("/__test/")) {
      res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(job));
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
          error: { code: "ANALYSIS_FAILED", message: "Анализ не завершён." },
        }),
      );
      return;
    }
    const event = {
      type: "Личные вещи на рабочем месте",
      timecode_sec: 1,
      description: "На рабочем столе видна личная вещь. Требуется проверка.",
      evidence_frames: [{ timestamp_sec: 1, image_base64: evidence.jpeg }],
      rule: {
        id: "R-PERSONAL-ITEMS",
        document: "Публичный источник",
        point: "п. 12",
        url: "https://adilet.zan.kz/rus/docs/V1900019680",
      },
    };
    res.end(
      JSON.stringify({
        job_id: jobId,
        status: "done",
        duration_sec: 5,
        verdict: job.name === "negative.mov" ? "no_violation" : "violation",
        violations:
          job.name === "negative.mov"
            ? []
            : job.name === "bad-result.mov"
              ? [{ ...event, timecode_sec: 8 }]
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
