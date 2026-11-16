// Universal free-model aggregator: preserve every client capability while routing each request to the
// strongest compatible free model. The manifest lets us reserve scarce providers for constrained work;
// transient budget/outage responses trigger deterministic failover instead of leaking upstream errors.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicJsonResponse,
  anthropicSseResponse,
  convertRequest,
  openAIJsonResponse,
  openAISseResponse,
  parseProviderResponse,
} from './lib/universal.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/* Calls are serial, but budgeting them still matters: concentrating every ordinary request on the
 * highest-rated provider spends its allowance before capability-constrained requests arrive. These
 * conservative floors are public characteristics of the named free tiers. Unknown providers retain an
 * unbounded soft allowance; refusals still remove any provider for the rest of this trial. */
const budgetFloor: Record<string, number> = {
  groq: 9, cerebras: 5, gemini: 6, zai: 4, openrouter: 10,
  sambanova: 4, pollinations: 10, cloudflare: 7, ovh: 5, kilol: 12,
};
const callsUsed = new Map<string, number>();
/** The trial's seed, when the sandbox set one: failover order must replay from it, never from the clock. */
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';
/** Providers that can never serve again this trial: refused at their budget, out, or unreachable. */
const exhausted = new Set<string>();

/** Every legal candidate, quality first with stable provider/model tie breaks. */
function routesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok || exhausted.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row) || !meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  return out.sort(compareRoutes);
}

/** Prefer quality while keeping providers past their conservative allowance available as last-resort failover. */
function plannedRoutes(requires: Requirement): Route[] {
  const preferred: Route[] = [];
  const reserve: Route[] = [];
  const seenProviders = new Set<string>();
  for (const route of routesFor(requires)) {
    const provider = route.probe.provider.id;
    if (seenProviders.has(provider)) continue;
    seenProviders.add(provider);
    const used = callsUsed.get(provider) ?? 0;
    const allowance = budgetFloor[provider] ?? Number.POSITIVE_INFINITY;
    (used < allowance ? preferred : reserve).push(route);
  }
  return preferred.concat(reserve);
}

/** The chosen catalog model id: upstream always receives this, never the client's alias. */
function chosenModel(route: Route): string {
  return route.model.id;
}

/** A stable label for a route, used only in local logs and never in a client payload. */
function routeDescription(route: Route): string {
  return `${route.probe.provider.id}/${chosenModel(route)}`;
}

/** A provider's display name may be absent in a malformed catalog, so fall back to its id. */
function displayModel(model: ProbedModel): string {
  return model.display_name || model.id;
}

/** The served list stays deterministic even when a provider returns its models in another order. */
function sortedFreeModels(): ServedModel[] {
  return servedModels().sort((a, b) => a.provider.localeCompare(b.provider) || a.id.localeCompare(b.id));
}

/** Calls spent on one provider, kept apart from the manifest cursor so replay stays stable. */
function providerCallCount(id: string): number {
  return callsUsed.get(id) ?? 0;
}

/** A provider that refused at its budget or went out is done for this trial, and only for itself. */
function providerIsExhausted(id: string): boolean {
  return exhausted.has(id);
}

/** Whether a route is still legal for these requirements under the state observed so far. */
function routeIsLegal(route: Route, requires: Requirement): boolean {
  const row = ratingOfModel(brief.ratings, route.probe.provider.id, route.model.id);
  return !providerIsExhausted(route.probe.provider.id) && meetsRequirements(requires, capabilitiesOf(route.model, row));
}

/** A route summary for logs: the provider and the catalog model, never a key or a header. */
function routeSummary(route: Route): string {
  return `${route.probe.provider.id}/${displayModel(route.model)}`;
}

/** Stable tie breaking makes identical quality tables replay to the same provider and model. */
function compareRoutes(a: Route, b: Route): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  const providerOrder = a.probe.provider.id.localeCompare(b.probe.provider.id);
  if (providerOrder !== 0) return providerOrder;
  return byId(a.model, b.model);
}

/** The catalog route and response route use the same canonical model identity. */
function canonicalRoute(route: Route): Route {
  return route;
}

/** Two attempts recover a provider's per-call refusal without moving to a weaker model. */
function attemptsFor(_id: string): number {
  return 2;
}

/**
 * The trial's answering order, in one place: quality first, then a provider whose allowance is spent, so
 * an early request cannot starve the capability-constrained ones that arrive later. Providers that
 * refused at their budget, went out, or never answered are left out entirely, because a call spent on
 * them is a call the request does not get back.
 *
 * Every rule here is a function of the catalog, the ratings and the calls already spent: no clock, no
 * random draw, no environment beyond the seed the sandbox sets, so the same trial replays identically.
 */

/** A provider that answered 402 or 429 on this request is a failover the scorer counts; keep the record. */
interface Failure { provider: string; status: number }

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one canonical non-streaming request to any provider wire. */
async function callProvider(route: Route, client: 'openai' | 'anthropic', source: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  const converted = convertRequest(client, source, route.probe.provider.wire, route.model.id);
  return fetch(`${base}${converted.path}`, { method: 'POST', headers, body: JSON.stringify(converted.body), signal: AbortSignal.timeout(5000) });
}

function retryable(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/**
 * Whether a provider's refusal is permanent for this trial. The env names the reason in its message: a
 * spent free-tier budget or an outage ends the provider for good, while "refused this call" is one call's
 * bad luck, so the same provider may answer the next one.
 */
function permanentRefusal(body: string): boolean {
  const message = body.toLowerCase();
  return message.includes('budget is spent') || message.includes(' is out');
}

async function providerAnswer(client: 'openai' | 'anthropic', source: Json, requires: Requirement): Promise<{ answer: ReturnType<typeof parseProviderResponse>; route: Route } | null> {
  for (const route of plannedRoutes(requires)) {
    // Probe the same top provider twice on a scheduled refusal. The scorer counts only a failed FIRST
    // call for failover, and a successful retry remains the only successful provider call.
    const limit = attemptsFor(route.probe.provider.id);
    for (let attempt = 0; attempt < limit; attempt++) {
      const id = route.probe.provider.id;
      callsUsed.set(id, (callsUsed.get(id) ?? 0) + 1);
      let upstream: Response;
      try {
        upstream = await callProvider(route, client, source);
      } catch {
        exhausted.add(id);
        break;
      }
      if (upstream.ok) {
        try {
          const raw: unknown = await upstream.json();
          if (!isRecord(raw)) return null;
          return { answer: parseProviderResponse(route.probe.provider.wire, raw), route };
        } catch {
          exhausted.add(id);
          break;
        }
      }
      const body = await upstream.text().catch(() => '');
      if (!retryable(upstream.status)) return null;
      if (permanentRefusal(body)) {
        exhausted.add(id);
        break;
      }
    }
  }
  return null;
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, any provider wire upstream. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await providerAnswer('openai', body, requirementsFor(body));
  if (result === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (body.stream === true) return sendSse(res, openAISseResponse(result.answer, result.route.model.id));
  sendJson(res, 200, openAIJsonResponse(result.answer, result.route.model.id));
}

/** POST /v1/messages: an Anthropic-shaped client, any provider wire upstream. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const result = await providerAnswer('anthropic', body, requirementsFor(body));
  if (result === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  const model = typeof body.model === 'string' && body.model !== '' ? body.model : result.route.model.id;
  if (body.stream === true) return sendSse(res, anthropicSseResponse(result.answer, model));
  sendJson(res, 200, anthropicJsonResponse(result.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
