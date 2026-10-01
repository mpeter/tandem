"""Authenticated OpenCode 2 HTTP admission and SSE execution lifecycle."""
from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import queue
import secrets
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ..commands import Command
from ..events import (ApprovalRequest, Failure, FileDiff, QuestionCancelled, QuestionRequest, TextDelta,
                      ThinkingDelta, ToolFinished, ToolOutput, ToolStarted,
                      TurnFinished, TurnOutcome)
from . import child_env, first_line, summarize_args
from .opencode import OpencodeRuntime, _free_port, mention_parts


def mention_files(prompt: str, cwd: str) -> list[dict]:
    files = []
    for part in mention_parts(prompt, cwd):
        path = (Path(cwd) / Path(part['source']['path']).expanduser()).resolve()
        if path.is_dir():
            uri = path.as_uri()
        else:
            mime = mimetypes.guess_type(path.name)[0] or 'text/plain'
            uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
        source = part['source']['text']
        files.append({'uri': uri, 'name': part['filename'], 'mention': {
            'start': source['start'], 'end': source['end'], 'text': source['value']}})
    return files


@dataclass
class TurnState:
    session_id: str
    inbox_id: str = ''
    delivered: bool = False
    status: str = ''
    error: str = ''
    tokens: dict = field(default_factory=dict)
    step_tokens: dict = field(default_factory=dict)
    tools: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    finished: set = field(default_factory=set)
    streams: dict = field(default_factory=dict)
    tool_output: dict = field(default_factory=dict)


class Opencode2HttpError(RuntimeError):
    def __init__(self, method, path, status, raw):
        super().__init__(f'{method} {path} -> {status}: {raw.decode(errors="replace")[:200]}')
        self.status = status
        try:
            self.data = json.loads(raw)
        except ValueError:
            self.data = None


class Opencode2Runtime(OpencodeRuntime):
    """Reuse only v1 process cleanup, subscription registration, and mention parsing."""
    def __init__(self, cfg, *, binary=None, base_url=None, password=None, turn_timeout=3600):
        super().__init__(cfg, binary=binary, base_url=base_url)
        self.password = password or secrets.token_urlsafe(32)
        self.turn_timeout = turn_timeout
        self._run_lock = threading.Lock()
        self._sse_conn = None
        self._inbox_id = None

    @property
    def _headers(self):
        auth = base64.b64encode(f'opencode:{self.password}'.encode()).decode()
        return {'Content-Type': 'application/json', 'Authorization': f'Basic {auth}'}

    def _http(self, method, path, body=None, timeout=30):
        u = urlparse(self.base_url)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
        try:
            conn.request(method, path, json.dumps(body).encode() if body is not None else None,
                         self._headers)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise Opencode2HttpError(method, path, resp.status, raw)
            return json.loads(raw) if raw else None
        finally:
            conn.close()

    def _healthy(self):
        try:
            return bool(self._http('GET', '/api/info', timeout=2))
        except (OSError, ValueError, RuntimeError, http.client.HTTPException):
            return False

    def ensure_server(self, cwd, health_timeout=10, *, tandem_id=None):
        if self._injected:
            return self.base_url
        with self._lock:
            if self._proc is not None and self._proc.poll() is None and self.base_url:
                return self.base_url
            port = _free_port()
            self.base_url = f'http://127.0.0.1:{port}'
            env = child_env(tandem_id=tandem_id)
            env['OPENCODE_PASSWORD'] = self.password
            self._proc = subprocess.Popen(
                [*self.binary, 'serve', '--hostname', '127.0.0.1', '--port', str(port)],
                cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, start_new_session=True)
            proc = self._proc
            self._sse_started = False

        def drain():
            try:
                for line in proc.stderr:
                    self._stderr.append(line.rstrip())
            finally:
                proc.stderr.close()

        self._drain = threading.Thread(target=drain, daemon=True)
        self._drain.start()
        deadline = time.monotonic() + health_timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._forget(proc)
                raise RuntimeError('opencode serve exited: ' + '\n'.join(self._stderr))
            if self._healthy():
                return self.base_url
            time.sleep(.1)
        self._forget(proc)
        raise RuntimeError(f'opencode serve did not become ready within {health_timeout:g}s')

    def _sse_reader(self, base, token):
        u = urlparse(base)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=None)
        self._sse_conn = conn
        try:
            conn.request('GET', '/api/event', headers=self._headers)
            resp = conn.getresponse()
            if resp.status != 200:
                raise RuntimeError(f'event stream returned {resp.status}')
            self._sse_ready.set()
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith('data:'):
                    self._events.put(json.loads(line[5:]))
            raise RuntimeError('event stream disconnected; completion events may have been lost')
        except Exception as exc:
            self._events.put({'type': '_connection_lost', 'data': {'error': str(exc)}})
        finally:
            conn.close()
            with self._lock:
                self._unregister(token)

    def _load_commands(self):
        # OpenCode 2.0.21's integration list awaits plugin activation; the
        # command list itself can return an incomplete cold-start catalog.
        try:
            activated = self._http('GET', '/api/integration')
            if not isinstance(activated, dict) or not isinstance(activated.get('data'), list):
                raise ValueError('invalid activation barrier response')
        except (OSError, ValueError, RuntimeError, http.client.HTTPException):
            raise RuntimeError('opencode plugin activation barrier failed; command catalog is unavailable') from None
        got = self._http('GET', '/api/command')
        self.harness_commands = [Command(c['name'], c.get('description', ''), 'opencode')
                                 for c in got['data']]

    def _check_local_history(self, native_id):
        if self._injected or not native_id:
            return
        from ...harness.opencode2 import database, db_path, require_fresh_session

        db = db_path()
        if db is None:
            raise RuntimeError('OpenCode 2 database is not discoverable; session history cannot be verified')
        with database(db) as conn:
            require_fresh_session(conn, native_id)

    def list_models(self, session):
        if not self._injected:
            self._check_local_history(session.native_id('opencode'))
        self.ensure_server(session.cwd, tandem_id=session.tandem_id)
        got = self._http('GET', '/api/model')
        return [f"{m['providerID']}/{m['id']}  {m.get('name', '')}".rstrip() for m in got['data']]

    def _busy(self, sid):
        active = self._http('GET', '/api/session/active')['data']
        inbox = self._http('GET', f'/api/session/{sid}/inbox')['data']
        # Agent switches can leave synthetic steering reminders for the next
        # prompt. An inactive reminder is context, rather than another turn.
        pending_work = any(item.get('type') != 'synthetic' or item.get('delivery') != 'steer'
                           for item in inbox)
        return sid in active or pending_work

    def _await_idle(self, sid, timeout=5):
        # The terminal event is published by the settled hook before the
        # coordinator removes its active registration.
        deadline = time.monotonic() + timeout
        while self._busy(sid):
            if time.monotonic() >= deadline:
                raise RuntimeError('opencode session did not become idle after execution ended')
            time.sleep(.05)

    def _form_answer(self, form, answers):
        result = {}
        for f in form['fields']:
            active = True
            for w in f.get('when', []):
                prior = result.get(w['key'])
                matches = w['value'] in prior if isinstance(prior, list) else prior == w['value']
                if w['key'] not in result or (matches if w['op'] == 'eq' else not matches) is False:
                    active = False
            if not active:
                continue
            if f.get('hidden'):
                if 'default' in f:
                    result[f['key']] = f['default']
                continue
            typ = f['type']
            if typ == 'external':
                raise RuntimeError(f"form requires external interaction: {f['url']}")
            options = f.get('options', [])
            labels = tuple(o['label'] for o in options)
            if typ == 'boolean':
                labels = ('true', 'false')
            title = f.get('title') or f['key']
            if typ == 'multiselect':
                title += ' (comma-separated choices)'
            if f.get('description'):
                title += '\n\n' + f['description']
            descriptions = [f"{o['label']}: {o['description']}" for o in options if o.get('description')]
            if descriptions:
                title += '\n\n' + '\n'.join(descriptions)
            raw = answers.answer(QuestionRequest(title, labels))
            values = {o['label']: o['value'] for o in options}
            if typ == 'boolean':
                if raw.lower() not in ('true', 'false'):
                    raise ValueError('boolean form answer must be true or false')
                value = raw.lower() == 'true'
            elif typ == 'integer':
                value = int(raw)
            elif typ == 'number':
                value = float(raw)
            elif typ == 'multiselect':
                value = [values.get(v.strip(), v.strip()) for v in raw.split(',') if v.strip()]
            elif typ == 'string':
                value = values.get(raw, raw)
            else:
                raise ValueError(f'unsupported form field {typ}')
            result[f['key']] = value
        return result

    def handle_event(self, ev, st, emit, answers):
        typ, data = ev.get('type', ''), ev.get('data') or {}
        if typ == '_connection_lost':
            raise RuntimeError(data['error'])
        form = data.get('form', {})
        if data.get('sessionID', form.get('sessionID')) != st.session_id:
            return
        sid = st.session_id
        if typ == '_cancelled':
            st.status = 'interrupted'
            return
        if typ == 'session.inbox.enqueued' and not st.inbox_id:
            # A command callback has no caller-supplied inbox ID. Correlate
            # user admission, rather than agent-switch steering reminders.
            if data.get('item', {}).get('type') == 'user':
                st.inbox_id = data['inboxID']
                self._inbox_id = st.inbox_id
        if typ == 'session.inbox.delivered' and data['inboxID'] == st.inbox_id:
            st.delivered = True
        if not st.delivered:
            return
        if typ.startswith('session.execution.'):
            terminal = typ.rsplit('.', 1)[1]
            if terminal in ('succeeded', 'failed', 'interrupted'):
                st.status = {'succeeded': 'completed', 'failed': 'failed', 'interrupted': 'interrupted'}[terminal]
                if terminal == 'failed':
                    st.error = str(data['error'].get('message') or data['error'])
                    emit(Failure(st.error))
        elif typ in ('session.text.delta', 'session.reasoning.delta'):
            key = (typ.split('.')[1], data['assistantMessageID'], data['ordinal'])
            st.streams[key] = st.streams.get(key, '') + data['delta']
            emit(ThinkingDelta(data['delta']) if 'reasoning' in typ else TextDelta(data['delta']))
        elif typ in ('session.text.ended', 'session.reasoning.ended'):
            key = (typ.split('.')[1], data['assistantMessageID'], data['ordinal'])
            previous, text = st.streams.get(key, ''), data['text']
            if text.startswith(previous) and len(text) > len(previous):
                emit(ThinkingDelta(text[len(previous):]) if 'reasoning' in typ else TextDelta(text[len(previous):]))
            st.streams[key] = text
        elif typ == 'session.step.ended':
            st.step_tokens[data.get('assistantMessageID', '')] = data['tokens']
            st.tokens = {key: sum(t.get(key, 0) for t in st.step_tokens.values())
                         for key in ('input', 'output')}
        elif typ == 'session.tool.input.started':
            st.tools[data['id']] = data['name']
        elif typ == 'session.tool.called':
            cid, inp = data['id'], data['input']
            tool = st.tools.get(cid, '')
            path = inp.get('filePath', inp.get('path', '')) if tool in ('edit', 'write') else ''
            st.paths[cid] = path
            emit(ToolStarted(cid, tool, summarize_args(tool, inp), (path,) if path else ()))
        elif typ == 'session.tool.progress':
            output = data['metadata'].get('output')
            if isinstance(output, str):
                self._tool_output(data['id'], output, st, emit)
        elif typ in ('session.tool.success', 'session.tool.failed'):
            cid = data['id']
            if cid in st.finished:
                return
            st.finished.add(cid)
            output = '\n'.join(c['text'] for c in data.get('content', []) if c['type'] == 'text')
            if output:
                self._tool_output(cid, output, st, emit)
            ok, error = typ.endswith('success'), data.get('error', {})
            emit(ToolFinished(cid, ok, '' if ok else first_line(str(error.get('message') or error))))
            diff = data.get('metadata', {}).get('diff')
            if isinstance(diff, str) and diff:
                emit(FileDiff(cid, st.paths.get(cid, ''), diff))
        elif typ == 'permission.asked':
            detail = data['action'] + ': ' + ', '.join(data['resources'])
            choice = answers.approve(ApprovalRequest('permission', first_line(detail)))
            self._http('POST', f"/api/session/{sid}/permission/{data['id']}/reply",
                       {'decision': {'allow': 'once', 'always': 'always'}.get(choice, 'reject')})
        elif typ == 'form.created':
            try:
                answer = self._form_answer(form, answers)
                self._http('POST', f"/api/session/{sid}/form/{form['id']}/reply", {'answer': answer})
            except QuestionCancelled:
                try:
                    self._http('DELETE', f"/api/session/{sid}/form/{form['id']}")
                except Opencode2HttpError as exc:
                    settled = {404: 'FormNotFoundError', 409: 'FormAlreadySettledError'}
                    if (exc.status not in settled or not isinstance(exc.data, dict)
                            or exc.data.get('_tag') != settled.get(exc.status)
                            or exc.data.get('id') != form['id']):
                        raise
            except Exception:
                self._http('DELETE', f"/api/session/{sid}/form/{form['id']}")
                raise

    @staticmethod
    def _tool_output(cid, text, st, emit):
        previous = st.tool_output.get(cid, '')
        delta = text[len(previous):] if text.startswith(previous) else text
        if delta:
            emit(ToolOutput(cid, delta))
        st.tool_output[cid] = text

    def run_turn(self, session, native_id, prompt, model, emit, answers, command=''):
        if not self._run_lock.acquire(blocking=False):
            error = 'opencode session already has an active Tandem turn'
            emit(Failure(error))
            emit(TurnFinished('failed'))
            return TurnOutcome('failed', error)
        st = TurnState(native_id)
        mode = self.cfg.effective_mode
        prior_agent, override = None, False
        self._interrupted = False
        try:
            if not native_id:
                raise ValueError('opencode sessions are created at pair time')
            selected_model = None
            if model:
                base, marker, variant = model.partition('#')
                if '/' not in base or not all(base.split('/', 1)) or (marker and not variant) or '#' in variant:
                    raise ValueError('opencode models are spelled provider/model[#variant]')
                provider, mid = base.split('/', 1)
                selected_model = {'providerID': provider, 'id': mid}
                if marker:
                    selected_model['variant'] = variant
            self._check_local_history(native_id)
            self.ensure_server(session.cwd, tandem_id=session.tandem_id)
            if self._busy(native_id):
                raise RuntimeError('opencode session is busy or has pending inbox input')
            self._load_commands()
            self._session_id = native_id
            self._turn_active.set()
            while not self._events.empty():
                self._events.get_nowait()
            self._start_sse()
            if not self._sse_ready.wait(10):
                raise RuntimeError('opencode event subscription did not become ready')
            prefix = f'/api/session/{native_id}'
            if mode == 'plan':
                prior_agent = self._http('GET', prefix)['data'].get('agent')
                if not prior_agent:
                    raise RuntimeError('plan requires an explicit prior agent selection to restore')
            # Without a session override, local tools inherit the foreground
            # server's environment, including its HTTP authentication password.
            variables = child_env(tandem_id=session.tandem_id)
            variables.pop('OPENCODE_PASSWORD', None)
            variables.pop('OPENCODE_SERVER_PASSWORD', None)
            self._http('PUT', prefix + '/environment', {'variables': variables})
            if mode == 'plan':
                override = True
                self._http('POST', prefix + '/agent', {'agent': 'plan'})
            if model:
                self._http('POST', prefix + '/model', {'model': selected_model})
            head = prompt.split(maxsplit=1)
            st.inbox_id = 'msg_' + uuid.uuid4().hex
            self._inbox_id = st.inbox_id
            if command == 'compact':
                path, body = prefix + '/compact', {'id': st.inbox_id}
            elif head and head[0].startswith('/') and head[0][1:] in {c.name for c in self.harness_commands}:
                st.inbox_id = ''
                self._inbox_id = None
                path, body = prefix + '/command', {'name': head[0][1:], 'text': head[1] if len(head) > 1 else ''}
            else:
                path, body = prefix + '/prompt', {'id': st.inbox_id, 'text': prompt}
            files = mention_files(prompt, session.cwd)
            if files and command != 'compact':
                body['files'] = files
            admitted = self._http('POST', path, body)
            if admitted is not None:
                st.inbox_id = admitted['data']['id']
                self._inbox_id = st.inbox_id
            deadline = time.monotonic() + self.turn_timeout
            while not st.status:
                if time.monotonic() >= deadline:
                    raise TimeoutError('opencode execution did not finish before the turn timeout')
                try:
                    ev = self._events.get(timeout=.1)
                except queue.Empty:
                    continue
                self.handle_event(ev, st, emit, answers)
            self._await_idle(native_id)
        except Exception as exc:
            st.status, st.error = 'failed', str(exc) or type(exc).__name__
            emit(Failure(st.error))
            if self._inbox_id:
                try:
                    self.interrupt()
                    self._await_idle(native_id)
                except Exception as stop_error:
                    emit(Failure(f'opencode interrupt failed: {stop_error}'))
        finally:
            if override:
                try:
                    self._http('POST', f'/api/session/{native_id}/agent', {'agent': prior_agent})
                    self._await_idle(native_id)
                except Exception as exc:
                    st.status, st.error = 'failed', f'opencode could not restore agent: {exc}'
                    emit(Failure(st.error))
            self._session_id = None
            self._inbox_id = None
            self._turn_active.clear()
            self._run_lock.release()
        usage = ''
        if 'input' in st.tokens and 'output' in st.tokens:
            usage = f"{st.tokens['input']}↑ {st.tokens['output']}↓"
        emit(TurnFinished(st.status, usage))
        return TurnOutcome(st.status, st.error)

    def interrupt(self):
        if self._session_id and self.base_url:
            self._interrupted = True
            sid = self._session_id
            response = self._http('POST', f'/api/session/{sid}/interrupt?resume=false', timeout=10)
            if not response['interrupted'] and self._inbox_id:
                self._http('DELETE', f'/api/session/{sid}/inbox/{self._inbox_id}')
                self._events.put({'type': '_cancelled', 'data': {'sessionID': sid}})

    def close(self):
        if self._sse_conn is not None and self._sse_conn.sock is not None:
            try:
                self._sse_conn.sock.shutdown(2)
            except OSError:
                pass
        super().close()
