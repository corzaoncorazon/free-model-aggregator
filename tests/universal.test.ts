import test from 'node:test';
import assert from 'node:assert/strict';

import {
  anthropicJsonResponse,
  anthropicSseResponse,
  convertRequest,
  openAIJsonResponse,
  openAISseResponse,
  parseProviderResponse,
} from '../lib/universal.ts';

const image = 'aGVsbG8=';
const openAI = {
  model: 'auto',
  stream: true,
  stream_options: { include_usage: true },
  messages: [
    { role: 'system', content: 'be exact' },
    { role: 'user', content: [{ type: 'text', text: 'first' }] },
    { role: 'assistant', content: 'ok' },
    { role: 'user', content: [{ type: 'text', text: 'last user' }, { type: 'image_url', image_url: { url: `data:image/png;base64,${image}` } }] },
  ],
  max_tokens: 77,
  temperature: 0.25,
  stop: ['END', 'HALT'],
  tools: [
    { type: 'function', function: { name: 'zeta', description: 'z', parameters: { type: 'object' } } },
    { type: 'function', function: { name: 'alpha', description: 'a', parameters: { type: 'object' } } },
  ],
  tool_choice: 'auto',
  response_format: { type: 'json_object' },
};

const anthropic = {
  model: 'auto',
  stream: true,
  system: [{ type: 'text', text: 'be exact' }],
  messages: [
    { role: 'user', content: 'first' },
    { role: 'assistant', content: [{ type: 'text', text: 'ok' }] },
    { role: 'user', content: [{ type: 'text', text: 'last user' }, { type: 'image', source: { type: 'base64', media_type: 'image/png', data: image } }] },
  ],
  max_tokens: 77,
  temperature: 0.25,
  stop_sequences: ['END', 'HALT'],
  tools: [
    { name: 'zeta', description: 'z', input_schema: { type: 'object' } },
    { name: 'alpha', description: 'a', input_schema: { type: 'object' } },
  ],
};

test('converts either client wire to OpenAI and forces non-streaming', () => {
  for (const [client, source] of [['openai', openAI], ['anthropic', anthropic]] as const) {
    const request = convertRequest(client, source, 'openai', 'chosen');
    assert.equal(request.path, '/v1/chat/completions');
    assert.equal(request.body.model, 'chosen');
    assert.equal(request.body.stream, false);
    assert.equal('stream_options' in request.body, false);
    assert.equal(request.body.max_tokens, 77);
    assert.deepEqual(request.body.stop, ['END', 'HALT']);
    assert.deepEqual((request.body.tools as any[]).map((x) => x.function.name), ['zeta', 'alpha']);
    assert.equal((request.body.messages as any[])[0].content, 'be exact');
    const last = (request.body.messages as any[]).at(-1);
    assert.equal(last.content[0].text, 'last user');
    assert.equal(last.content[1].image_url.url, `data:image/png;base64,${image}`);
  }
});

test('converts either client wire to Anthropic and preserves echo fields', () => {
  for (const [client, source] of [['openai', openAI], ['anthropic', anthropic]] as const) {
    const request = convertRequest(client, source, 'anthropic', 'chosen');
    assert.equal(request.path, '/v1/messages');
    assert.equal(request.body.model, 'chosen');
    assert.equal(request.body.stream, false);
    assert.equal(request.body.max_tokens, 77);
    assert.deepEqual(request.body.stop_sequences, ['END', 'HALT']);
    assert.deepEqual((request.body.tools as any[]).map((x) => x.name), ['zeta', 'alpha']);
    assert.equal(request.body.system, 'be exact');
    const last = (request.body.messages as any[]).at(-1);
    assert.equal(last.content[0].text, 'last user');
    assert.deepEqual(last.content[1].source, { type: 'base64', media_type: 'image/png', data: image });
  }
});

test('converts either client wire to Gemini preserving images, tools and generation config', () => {
  for (const [client, source] of [['openai', openAI], ['anthropic', anthropic]] as const) {
    const request = convertRequest(client, source, 'gemini', 'gemini/x y');
    assert.equal(request.path, '/v1beta/models/gemini%2Fx%20y:generateContent');
    assert.equal(request.body.stream, undefined);
    assert.deepEqual(request.body.systemInstruction, { parts: [{ text: 'be exact' }] });
    assert.deepEqual((request.body.tools as any[])[0].functionDeclarations.map((x: any) => x.name), ['zeta', 'alpha']);
    assert.deepEqual(request.body.generationConfig, {
      maxOutputTokens: 77,
      temperature: 0.25,
      stopSequences: ['END', 'HALT'],
      ...(client === 'openai' ? { responseMimeType: 'application/json' } : {}),
    });
    const last = (request.body.contents as any[]).at(-1);
    assert.equal(last.role, 'user');
    assert.equal(last.parts[0].text, 'last user');
    assert.deepEqual(last.parts[1].inlineData, { mimeType: 'image/png', data: image });
  }
});

test('parses all provider non-stream responses to one canonical result', () => {
  const expected = { text: 'α\nβ', inputTokens: 11, outputTokens: 7, stop: 'stop' };
  assert.deepEqual(parseProviderResponse('openai', {
    choices: [{ message: { content: 'α\nβ' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 11, completion_tokens: 7 },
  }), expected);
  assert.deepEqual(parseProviderResponse('anthropic', {
    content: [{ type: 'text', text: 'α' }, { type: 'text', text: '\nβ' }],
    stop_reason: 'end_turn', usage: { input_tokens: 11, output_tokens: 7 },
  }), { ...expected, stop: 'end_turn' });
  assert.deepEqual(parseProviderResponse('gemini', {
    candidates: [{ content: { parts: [{ text: 'α' }, { text: '\nβ' }] }, finishReason: 'STOP' }],
    usageMetadata: { promptTokenCount: 11, candidatesTokenCount: 7 },
  }), { ...expected, stop: 'STOP' });
});

test('synthesizes OpenAI JSON and SSE with exact text, usage and mapped stop family', () => {
  const answer = { text: 'exact 😃\ntext', inputTokens: 11, outputTokens: 7, stop: 'max_tokens' };
  const json = openAIJsonResponse(answer, 'client-model');
  assert.equal((json.choices as any[])[0].message.content, answer.text);
  assert.equal((json.choices as any[])[0].finish_reason, 'length');
  assert.deepEqual(json.usage, { prompt_tokens: 11, completion_tokens: 7, total_tokens: 18 });
  const sse = openAISseResponse(answer, 'client-model');
  const frames = sse.trim().split('\n\n').map((f) => f.slice(6));
  assert.equal(frames.at(-1), '[DONE]');
  const chunks = frames.slice(0, -1).map(JSON.parse);
  assert.equal(chunks.map((x) => x.choices[0]?.delta?.content ?? '').join(''), answer.text);
  assert.equal(chunks.find((x) => x.choices[0]?.finish_reason)?.choices[0].finish_reason, 'length');
  assert.deepEqual(chunks.find((x) => x.choices.length === 0).usage, json.usage);
});

test('synthesizes Anthropic JSON and ordered SSE with exact text, usage and mapped stop family', () => {
  const answer = { text: 'exact 😃\ntext', inputTokens: 11, outputTokens: 7, stop: 'tool_calls' };
  const json = anthropicJsonResponse(answer, 'client-model');
  assert.equal((json.content as any[])[0].text, answer.text);
  assert.equal(json.stop_reason, 'tool_use');
  assert.deepEqual(json.usage, { input_tokens: 11, output_tokens: 7 });
  const frames = anthropicSseResponse(answer, 'client-model').trim().split('\n\n');
  assert.deepEqual(frames.map((f) => f.split('\n')[0]), [
    'event: message_start', 'event: content_block_start', 'event: content_block_delta',
    'event: content_block_stop', 'event: message_delta', 'event: message_stop',
  ]);
  const data = frames.map((f) => JSON.parse(f.split('\n').find((x) => x.startsWith('data: '))!.slice(6)));
  assert.equal(data[2].delta.text, answer.text);
  assert.equal(data[4].delta.stop_reason, 'tool_use');
  assert.equal(data[0].message.usage.input_tokens, 11);
  assert.equal(data[4].usage.output_tokens, 7);
});
