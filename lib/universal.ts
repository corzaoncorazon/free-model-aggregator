import { isRecord, type Json, type Wire } from './providers.ts';

export interface CanonicalAnswer { text: string; inputTokens: number; outputTokens: number; stop: string }
export interface ProviderRequest { path: string; body: Json }
type ClientWire = 'openai' | 'anthropic';

const arr = (v: unknown): unknown[] => Array.isArray(v) ? v : [];
const num = (v: unknown): number => typeof v === 'number' && Number.isFinite(v) ? v : 0;
const strings = (v: unknown): string[] => typeof v === 'string' ? [v] : arr(v).filter((x): x is string => typeof x === 'string');
function answerText(v: unknown): string {
  if (typeof v === 'string') return v;
  return arr(v).flatMap(x => isRecord(x) && x.type === 'text' && typeof x.text === 'string' ? [x.text] : []).join('');
}
function textBlocks(v: unknown): string {
  if (typeof v === 'string') return v;
  return arr(v).flatMap(x => isRecord(x) && x.type === 'text' && typeof x.text === 'string' ? [x.text] : []).join('\n');
}
function dataImage(url: unknown): { media: string; data: string } | null {
  if (typeof url !== 'string') return null;
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  return m ? { media: m[1], data: m[2] } : null;
}

interface Normalized {
  system: string; messages: Json[]; max: number | null; temperature: number | null;
  stops: string[]; tools: Json[]; json: boolean;
}
function normalize(client: ClientWire, body: Json): Normalized {
  if (client === 'anthropic') {
    return {
      system: textBlocks(body.system),
      messages: arr(body.messages).filter(isRecord),
      max: typeof body.max_tokens === 'number' ? body.max_tokens : null,
      temperature: typeof body.temperature === 'number' ? body.temperature : null,
      stops: strings(body.stop_sequences),
      tools: arr(body.tools).filter(isRecord).map(t => ({ name: typeof t.name === 'string' ? t.name : '', description: typeof t.description === 'string' ? t.description : '', schema: isRecord(t.input_schema) ? t.input_schema : {} })),
      json: false,
    };
  }
  const messages = arr(body.messages).filter(isRecord);
  const system = messages.filter(m => m.role === 'system').map(m => textBlocks(m.content)).filter(Boolean).join('\n');
  return {
    system, messages: messages.filter(m => m.role !== 'system'),
    max: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    stops: strings(body.stop),
    tools: arr(body.tools).filter(isRecord).flatMap(t => isRecord(t.function) ? [{ name: typeof t.function.name === 'string' ? t.function.name : '', description: typeof t.function.description === 'string' ? t.function.description : '', schema: isRecord(t.function.parameters) ? t.function.parameters : {} }] : []),
    json: isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'),
  };
}
function openAIParts(content: unknown): unknown {
  if (typeof content === 'string') return content;
  return arr(content).flatMap((x): Json[] => {
    if (!isRecord(x)) return [];
    if (x.type === 'text' && typeof x.text === 'string') return [{ type: 'text', text: x.text }];
    if (x.type === 'image_url' && isRecord(x.image_url) && typeof x.image_url.url === 'string') return [{ type: 'image_url', image_url: { url: x.image_url.url } }];
    if (x.type === 'image' && isRecord(x.source)) {
      if (x.source.type === 'base64' && typeof x.source.data === 'string') return [{ type: 'image_url', image_url: { url: `data:${typeof x.source.media_type === 'string' ? x.source.media_type : 'image/png'};base64,${x.source.data}` } }];
      if (x.source.type === 'url' && typeof x.source.url === 'string') return [{ type: 'image_url', image_url: { url: x.source.url } }];
    }
    return [];
  });
}
function anthropicParts(content: unknown): unknown {
  if (typeof content === 'string') return content;
  return arr(content).flatMap((x): Json[] => {
    if (!isRecord(x)) return [];
    if (x.type === 'text' && typeof x.text === 'string') return [{ type: 'text', text: x.text }];
    if (x.type === 'image' && isRecord(x.source)) return [{ type: 'image', source: x.source }];
    if (x.type === 'image_url' && isRecord(x.image_url)) {
      const d = dataImage(x.image_url.url);
      if (d) return [{ type: 'image', source: { type: 'base64', media_type: d.media, data: d.data } }];
      if (typeof x.image_url.url === 'string') return [{ type: 'image', source: { type: 'url', url: x.image_url.url } }];
    }
    return [];
  });
}
function geminiParts(content: unknown): Json[] {
  if (typeof content === 'string') return [{ text: content }];
  return arr(content).flatMap((x): Json[] => {
    if (!isRecord(x)) return [];
    if (x.type === 'text' && typeof x.text === 'string') return [{ text: x.text }];
    if (x.type === 'image' && isRecord(x.source)) {
      if (x.source.type === 'base64' && typeof x.source.data === 'string') return [{ inlineData: { mimeType: typeof x.source.media_type === 'string' ? x.source.media_type : 'image/png', data: x.source.data } }];
      if (x.source.type === 'url' && typeof x.source.url === 'string') return [{ fileData: { fileUri: x.source.url } }];
    }
    if (x.type === 'image_url' && isRecord(x.image_url)) {
      const d = dataImage(x.image_url.url);
      if (d) return [{ inlineData: { mimeType: d.media, data: d.data } }];
      if (typeof x.image_url.url === 'string') return [{ fileData: { fileUri: x.image_url.url } }];
    }
    return [];
  });
}

export function convertRequest(client: ClientWire, source: Json, provider: Wire, model: string): ProviderRequest {
  const n = normalize(client, source);
  if (provider === 'openai') {
    const messages: Json[] = [];
    if (n.system) messages.push({ role: 'system', content: n.system });
    for (const m of n.messages) messages.push({ role: m.role === 'assistant' ? 'assistant' : 'user', content: openAIParts(m.content) });
    const body: Json = { model, messages, stream: false };
    if (n.max !== null) body.max_tokens = n.max;
    if (n.temperature !== null) body.temperature = n.temperature;
    if (n.stops.length) body.stop = n.stops;
    if (n.tools.length) body.tools = n.tools.map(t => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } }));
    if (n.json) body.response_format = { type: 'json_object' };
    return { path: '/v1/chat/completions', body };
  }
  if (provider === 'anthropic') {
    const messages = n.messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: anthropicParts(m.content) }));
    const body: Json = { model, messages, max_tokens: n.max ?? 1024, stream: false };
    if (n.system) body.system = n.system;
    if (n.temperature !== null) body.temperature = n.temperature;
    if (n.stops.length) body.stop_sequences = n.stops;
    if (n.tools.length) body.tools = n.tools.map(t => ({ name: t.name, description: t.description, input_schema: t.schema }));
    return { path: '/v1/messages', body };
  }
  const contents = n.messages.map(m => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: geminiParts(m.content) }));
  const config: Json = {};
  if (n.max !== null) config.maxOutputTokens = n.max;
  if (n.temperature !== null) config.temperature = n.temperature;
  if (n.stops.length) config.stopSequences = n.stops;
  if (n.json) config.responseMimeType = 'application/json';
  const body: Json = { contents };
  if (n.system) body.systemInstruction = { parts: [{ text: n.system }] };
  if (n.tools.length) body.tools = [{ functionDeclarations: n.tools.map(t => ({ name: t.name, description: t.description, parameters: t.schema })) }];
  if (Object.keys(config).length) body.generationConfig = config;
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body };
}

export function parseProviderResponse(wire: Wire, body: Json): CanonicalAnswer {
  if (wire === 'openai') {
    const choice = isRecord(arr(body.choices)[0]) ? arr(body.choices)[0] as Json : {};
    const msg = isRecord(choice.message) ? choice.message : {};
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: typeof msg.content === 'string' ? msg.content : textBlocks(msg.content), inputTokens: num(usage.prompt_tokens), outputTokens: num(usage.completion_tokens), stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
  }
  if (wire === 'anthropic') {
    const usage = isRecord(body.usage) ? body.usage : {};
    return { text: answerText(body.content), inputTokens: num(usage.input_tokens), outputTokens: num(usage.output_tokens), stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn' };
  }
  const candidate = isRecord(arr(body.candidates)[0]) ? arr(body.candidates)[0] as Json : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const text = arr(content.parts).flatMap(x => isRecord(x) && typeof x.text === 'string' ? [x.text] : []).join('');
  return { text, inputTokens: num(usage.promptTokenCount), outputTokens: num(usage.candidatesTokenCount), stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP' };
}

function openAIStop(s: string): string { if (s === 'max_tokens' || s === 'MAX_TOKENS' || s === 'length') return 'length'; if (s === 'tool_use' || s === 'tool_calls' || s === 'function_call') return 'tool_calls'; if (s === 'SAFETY' || s === 'refusal' || s === 'content_filter') return 'content_filter'; return 'stop'; }
function anthropicStop(s: string): string { if (s === 'max_tokens' || s === 'MAX_TOKENS' || s === 'length') return 'max_tokens'; if (s === 'tool_use' || s === 'tool_calls' || s === 'function_call') return 'tool_use'; if (s === 'SAFETY' || s === 'refusal' || s === 'content_filter') return 'refusal'; return 'end_turn'; }
const usageOpenAI = (a: CanonicalAnswer) => ({ prompt_tokens: a.inputTokens, completion_tokens: a.outputTokens, total_tokens: a.inputTokens + a.outputTokens });
export function openAIJsonResponse(a: CanonicalAnswer, model: string): Json { return { id: 'chatcmpl-aggregate', object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message: { role: 'assistant', content: a.text }, finish_reason: openAIStop(a.stop) }], usage: usageOpenAI(a) }; }
const frame = (data: unknown) => `data: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`;
export function openAISseResponse(a: CanonicalAnswer, model: string): string {
  const base = { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: 1767225600, model };
  return frame({ ...base, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] }) + frame({ ...base, choices: [{ index: 0, delta: { content: a.text }, finish_reason: null }] }) + frame({ ...base, choices: [{ index: 0, delta: {}, finish_reason: openAIStop(a.stop) }] }) + frame({ ...base, choices: [], usage: usageOpenAI(a) }) + frame('[DONE]');
}
export function anthropicJsonResponse(a: CanonicalAnswer, model: string): Json { return { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [{ type: 'text', text: a.text }], stop_reason: anthropicStop(a.stop), stop_sequence: null, usage: { input_tokens: a.inputTokens, output_tokens: a.outputTokens } }; }
const event = (name: string, data: unknown) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
export function anthropicSseResponse(a: CanonicalAnswer, model: string): string {
  return event('message_start', { type: 'message_start', message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: a.inputTokens, output_tokens: 0 } } }) + event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }) + event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: a.text } }) + event('content_block_stop', { type: 'content_block_stop', index: 0 }) + event('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStop(a.stop), stop_sequence: null }, usage: { output_tokens: a.outputTokens } }) + event('message_stop', { type: 'message_stop' });
}