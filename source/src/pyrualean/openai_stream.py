"""Chat-completions SSE transport: assemble complete replies before executing tools."""
import json
import threading
import time

import httpx


def assemble(lines, emit, deadline, stopped=lambda: False):
    message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
    calls, usage, finish = {}, {}, None
    first = set()
    counts = {'content_chars': 0, 'reasoning_chars': 0, 'argument_chars': 0}
    last_progress = time.monotonic()
    data = []

    def packets():
        for line in lines:
            if stopped() or time.monotonic() >= deadline:
                raise TimeoutError('Streaming request stopped or exceeded deadline')
            if line == '':
                if data:
                    yield '\n'.join(data)
                    data.clear()
            elif line.startswith('data:'):
                data.append(line[5:].lstrip())
        if data:
            yield '\n'.join(data)

    for raw in packets():
        if raw == '[DONE]':
            if finish is None:
                raise ValueError('Stream ended without finish_reason; partial reply discarded')
            if calls:
                message['tool_calls'] = [calls[i] for i in sorted(calls)]
            return {'choices': [{'message': message, 'finish_reason': finish}], 'usage': usage}
        item = json.loads(raw)
        if item.get('error'):
            raise ValueError('Provider stream error: ' + str(item['error']))
        if item.get('usage'):
            usage = item['usage']
        for choice in item.get('choices', []):
            if choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta', {})
            for key, counter in [('content', 'content_chars'), ('reasoning_content', 'reasoning_chars')]:
                value = delta.get(key) or ''
                if value:
                    if key not in first:
                        emit('stream_first_delta', channel=key)
                        first.add(key)
                    message[key] += value
                    counts[counter] += len(value)
            for part in delta.get('tool_calls', []):
                index = part['index']
                call = calls.setdefault(index, {'id': '', 'type': 'function',
                                                'function': {'name': '', 'arguments': ''}})
                if part.get('id'):
                    call['id'] = part['id']
                fn = part.get('function', {})
                for key in ('name', 'arguments'):
                    call['function'][key] += fn.get(key) or ''
                counts['argument_chars'] += len(fn.get('arguments') or '')
                if 'tool_calls' not in first:
                    emit('stream_first_delta', channel='tool_calls')
                    first.add('tool_calls')
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
        if time.monotonic() - last_progress >= 5:
            emit('stream_progress', **counts)
            last_progress = time.monotonic()
    raise ValueError('Incomplete SSE stream: missing [DONE]; partial reply discarded')


def streaming_transport(api_key, event, should_stop=None):
    """Each attempt has a wall deadline, including continuously arriving tokens."""
    def post(url, body, timeout_s):
        start = time.monotonic()
        def emit(kind, **fields):
            event(kind, latency_s=round(time.monotonic() - start, 3), **fields)
        payload = {**body, 'stream': True,
                   'stream_options': {**body.get('stream_options', {}), 'include_usage': True}}
        # Closing the per-request client also interrupts a blocked socket read.
        with httpx.Client(headers={'Authorization': 'Bearer ' + api_key} if api_key else {},
                          timeout=timeout_s, trust_env=True) as client:
            timer = threading.Timer(timeout_s, client.close)
            timer.daemon = True
            timer.start()
            try:
                with client.stream('POST', url, json=payload) as response:
                    emit('stream_headers', status_code=response.status_code)
                    if response.status_code != 200:
                        return response.status_code, response.read().decode(errors='replace')
                    if 'text/event-stream' not in response.headers.get('content-type', ''):
                        emit('stream_fallback', reason='server_returned_nonstream_response')
                        return response.status_code, json.loads(response.read())
                    result = assemble(response.iter_lines(), emit, start + timeout_s,
                                      should_stop or (lambda: False))
                    emit('stream_complete')
                    return 200, result
            except Exception as exc:
                if time.monotonic() >= start + timeout_s:
                    emit('stream_timeout', reason='total_request_deadline')
                    raise TimeoutError('Streaming request exceeded total request deadline') from exc
                raise
            finally:
                timer.cancel()
    return post
