"""HTTP/SSE transport contract for the OpenCode 2 runtime."""
import base64
import io
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tandem.chat.events import (Failure, QuestionRequest, TextDelta, ToolFinished,
                                ToolOutput, ToolStarted, TurnFinished)
from tandem.chat.render import Screen
from tandem.chat.runtime.opencode2 import Opencode2Runtime, TurnState, mention_files
from tandem.config import ChatConfig


SID = 'ses_test'
SESSION = SimpleNamespace(cwd='/tmp', tandem_id='tdm-test')
PLAN_REMINDER = {
    'id': 'msg_restore', 'sessionID': SID, 'type': 'synthetic', 'delivery': 'steer',
    'time': {'created': 1},
    'payload': {'text': 'You are no longer in Plan mode.', 'description': 'Agent changed'},
}


class Recorder:
    def __init__(self, answer='Blue', approve='always'):
        self.events, self.questions, self.approvals = [], [], []
        self.answer_value, self.approve_value = answer, approve

    def emit(self, event): self.events.append(event)
    def answer(self, req): self.questions.append(req); return self.answer_value
    def approve(self, req): self.approvals.append(req); return self.approve_value


class FakeServer:
    def __init__(self):
        self.calls, self.headers = [], []
        self.environments = []
        self.events = queue.Queue()
        self.posted = threading.Event()
        self.active, self.pending = {}, []
        self.agent = 'build'
        self.admission_error = False
        self.environment_error = False
        self.restore_delay = 0
        self.restore_settled = threading.Event()
        self.restore_reminder = None
        self.form_cancel_error = None
        self.form_cancel_after_interrupt = False
        self.interrupted = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args): pass

            def send_json(self, body, code=200):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                fake.headers.append(self.headers.get('Authorization'))
                if self.path == '/api/event':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    self.wfile.flush()
                    while True:
                        event = fake.events.get()
                        if event is None:
                            self.close_connection = True
                            return
                        try:
                            self.wfile.write(('data: ' + json.dumps(event) + '\n\n').encode())
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                elif self.path == '/api/info':
                    self.send_json({'version': '2.0.21'})
                elif self.path == '/api/session/active':
                    self.send_json({'data': fake.active})
                elif self.path.endswith('/inbox'):
                    self.send_json({'data': fake.pending})
                elif self.path == '/api/command':
                    self.send_json({'location': {'directory': '/tmp'}, 'data': [{'name': 'explain'}]})
                elif self.path == '/api/model':
                    self.send_json({'location': {'directory': '/tmp'}, 'data': [
                        {'providerID': 'openai', 'id': 'gpt', 'name': 'GPT'}]})
                elif self.path == f'/api/session/{SID}':
                    self.send_json({'data': {'agent': fake.agent}})
                else:
                    self.send_json({'error': self.path}, 404)

            def do_POST(self):
                fake.headers.append(self.headers.get('Authorization'))
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or 'null')
                fake.calls.append((self.path, body))
                if self.path.endswith(('/prompt', '/compact', '/command')):
                    if fake.admission_error:
                        self.send_json({'message': 'bad prompt'}, 400)
                        return
                    inbox_id = body.get('id', 'msg_command')
                    self.send_json({'data': {'id': inbox_id}} if not self.path.endswith('/command') else None)
                    fake.posted.set()
                elif '/interrupt?' in self.path:
                    fake.interrupted.set()
                    fake.publish('session.execution.interrupted', reason='user')
                    self.send_json({'interrupted': True})
                elif self.path.endswith('/agent') and body['agent'] == 'build' and fake.restore_delay:
                    fake.active = {SID: {'type': 'running'}}
                    fake.pending = [fake.restore_reminder or {'id': 'msg_restore', 'type': 'synthetic'}]

                    def settle_restore():
                        time.sleep(fake.restore_delay)
                        fake.active = {}
                        fake.pending = [fake.restore_reminder] if fake.restore_reminder else []
                        fake.restore_settled.set()

                    threading.Thread(target=settle_restore, daemon=True).start()
                    self.send_json(None)
                else:
                    self.send_json(None)

            def do_PUT(self):
                fake.headers.append(self.headers.get('Authorization'))
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                fake.environments.append((self.path, body, len(fake.calls)))
                self.send_json({'message': 'environment rejected'} if fake.environment_error else None,
                               403 if fake.environment_error else 200)

            def do_DELETE(self):
                fake.calls.append((self.path, None))
                if '/form/' in self.path:
                    if fake.form_cancel_after_interrupt and not fake.interrupted.wait(2):
                        self.send_json({'message': 'interrupt did not settle first'}, 500)
                        return
                    if fake.form_cancel_error:
                        status, error = fake.form_cancel_error
                        self.send_json(error, status)
                        return
                self.send_json(None)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'

    def publish(self, typ, **data):
        self.events.put({'type': typ, 'data': {'sessionID': SID, **data}})

    def deliver(self):
        path, body = next(c for c in self.calls if c[0].endswith(('/prompt', '/command', '/compact')))
        inbox = body.get('id', 'msg_command')
        item = {'type': 'compaction' if path.endswith('/compact') else 'user',
                'delivery': 'steer',
                'payload': {} if path.endswith('/compact') else {'text': body['text']}}
        self.publish('session.inbox.enqueued', inboxID=inbox, item=item)
        self.publish('session.inbox.delivered', inboxID=inbox)

    def stop(self):
        self.events.put(None)
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake():
    server = FakeServer()
    yield server
    server.stop()


def start_turn(fake, cfg=None, *, prompt='hi', command='', model=''):
    rt = Opencode2Runtime(cfg or ChatConfig(), base_url=fake.base_url,
                          password='private-test-password', turn_timeout=.8)
    rec, result = Recorder(), []
    worker = threading.Thread(target=lambda: result.append(
        rt.run_turn(SESSION, SID, prompt, model, rec.emit, rec, command)))
    worker.start()
    assert fake.posted.wait(2)
    return rt, rec, result, worker


def finish(worker, rt):
    worker.join(3)
    assert not worker.is_alive()
    rt.close()


def test_admission_is_not_completion_and_unrelated_terminal_is_ignored(fake):
    rt, rec, result, worker = start_turn(fake, model='openai/gpt')
    assert result == []
    fake.publish('session.execution.succeeded')  # stale terminal before delivery
    fake.publish('session.inbox.delivered', inboxID='msg_other')
    time.sleep(.05)
    assert result == []
    fake.deliver()
    fake.publish('session.execution.succeeded', sessionID='ses_other')
    fake.publish('session.text.delta', assistantMessageID='msg_a', ordinal=0, delta='Hi')
    fake.publish('session.text.ended', assistantMessageID='msg_a', ordinal=0, text='Hi!')
    fake.publish('session.step.ended', tokens={'input': 12, 'output': 3})
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert result[0].status == 'completed'
    assert rec.events == [TextDelta('Hi'), TextDelta('!'), TurnFinished('completed', '12↑ 3↓')]
    assert fake.calls[0] == (f'/api/session/{SID}/model', {'model': {'providerID': 'openai', 'id': 'gpt'}})
    auth = 'Basic ' + base64.b64encode(b'opencode:private-test-password').decode()
    assert fake.headers and all(h == auth for h in fake.headers)


@pytest.mark.parametrize('terminal,status', [('succeeded', 'completed'), ('failed', 'failed'),
                                            ('interrupted', 'interrupted')])
def test_plan_restores_previous_agent_after_every_terminal(fake, terminal, status):
    rt, rec, result, worker = start_turn(fake, ChatConfig(mode='plan'))
    fake.deliver()
    fake.publish('session.execution.' + terminal, error={'message': 'provider failed'}, reason='user')
    finish(worker, rt)
    assert result[0].status == status
    agents = [body for path, body in fake.calls if path.endswith('/agent')]
    assert agents == [{'agent': 'plan'}, {'agent': 'build'}]


def test_interrupt_uses_v2_route_and_waits_for_terminal(fake):
    rt, rec, result, worker = start_turn(fake, ChatConfig(mode='plan'))
    fake.deliver()
    fake.publish('session.text.delta', assistantMessageID='msg_a', ordinal=0, delta='started')
    deadline = time.monotonic() + 1
    while not rec.events and time.monotonic() < deadline:
        time.sleep(.01)
    rt.interrupt()
    finish(worker, rt)
    assert result[0].status == 'interrupted'
    assert (f'/api/session/{SID}/interrupt?resume=false', None) in fake.calls
    assert fake.calls[-1] == (f'/api/session/{SID}/agent', {'agent': 'build'})


def test_disconnect_fails_and_restores_plan(fake):
    rt, rec, result, worker = start_turn(fake, ChatConfig(mode='plan'))
    fake.deliver()
    fake.events.put(None)
    finish(worker, rt)
    assert result[0].status == 'failed'
    assert 'disconnected' in result[0].error
    assert fake.calls[-1] == (f'/api/session/{SID}/agent', {'agent': 'build'})


def test_permissions_forms_and_tools_use_actual_schemas(fake):
    rt, rec, result, worker = start_turn(fake)
    fake.deliver()
    fake.publish('permission.asked', id='per_x', action='bash', resources=['echo ok'])
    fake.publish('form.created', form={'id': 'frm_x', 'sessionID': SID, 'title': 'Color', 'fields': [
        {'key': 'color', 'type': 'string', 'title': 'Which color?',
         'options': [{'label': 'Blue', 'value': 'blue_id'}]}]})
    fake.publish('session.tool.input.started', id='tool_x', name='bash')
    fake.publish('session.tool.called', id='tool_x', input={'command': 'echo ok'})
    fake.publish('session.tool.success', id='tool_x', content=[{'type': 'text', 'text': 'ok'}])
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert result[0].status == 'completed'
    assert (f'/api/session/{SID}/permission/per_x/reply', {'decision': 'always'}) in fake.calls
    assert (f'/api/session/{SID}/form/frm_x/reply', {'answer': {'color': 'blue_id'}}) in fake.calls
    assert rec.questions == [QuestionRequest('Which color?', ('Blue',))]
    assert ToolStarted('tool_x', 'bash', 'echo ok') in rec.events
    assert ToolOutput('tool_x', 'ok') in rec.events
    assert ToolFinished('tool_x', True) in rec.events


@pytest.mark.parametrize('field_type,reply,expected', [
    ('string', 'SQLite', 'sqlite_id'),
    ('multiselect', 'SQLite, Memory', ['sqlite_id', 'memory_id']),
])
def test_native_question_details_render_without_changing_answer_values(fake, field_type, reply, expected):
    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    rec = Recorder(answer=reply)
    form = {'id': 'frm_details', 'sessionID': SID, 'title': 'Questions', 'fields': [
        {'key': 'storage', 'type': field_type, 'title': 'Storage',
         'description': 'Where should generated records be stored?',
         'options': [
             {'label': 'Memory', 'value': 'memory_id',
              'description': 'Discard records when the process exits.'},
             {'label': 'SQLite', 'value': 'sqlite_id',
              'description': 'Persist records across process restarts.'}]}]}
    rt.handle_event({'type': 'form.created', 'data': {'form': form}},
                    TurnState(SID, delivered=True), rec.emit, rec)
    req, = rec.questions
    assert req.options == ('Memory', 'SQLite')
    assert 'Where should generated records be stored?' in req.prompt
    assert 'Memory: Discard records when the process exits.' in req.prompt
    assert 'SQLite: Persist records across process restarts.' in req.prompt
    if field_type == 'multiselect':
        assert '(comma-separated choices)' in req.prompt
    chunks = []
    screen = Screen(chunks.append, rows=24, cols=100, cfg=ChatConfig(), color=False)
    screen.question(req)
    rendered = b''.join(chunks).decode()
    assert 'Where should generated records be stored?' in rendered
    assert 'Memory: Discard records when the process exits.' in rendered
    assert 'SQLite: Persist records across process restarts.' in rendered
    assert '1. Memory' in rendered and '2. SQLite' in rendered
    assert fake.calls == [(f'/api/session/{SID}/form/frm_details/reply',
                           {'answer': {'storage': expected}})]


def test_cancelling_question_with_details_deletes_form_without_reply(fake):
    class CancelledAnswer(Recorder):
        def answer(self, req):
            self.questions.append(req)
            raise RuntimeError('question cancelled')

    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    rec = CancelledAnswer()
    form = {'id': 'frm_cancel', 'sessionID': SID, 'fields': [
        {'key': 'storage', 'type': 'string', 'title': 'Storage',
         'description': 'Where should generated records be stored?',
         'options': [{'label': 'Memory', 'value': 'memory_id',
                      'description': 'Discard records when the process exits.'}]}]}
    with pytest.raises(RuntimeError, match='question cancelled'):
        rt.handle_event({'type': 'form.created', 'data': {'form': form}},
                        TurnState(SID, delivered=True), rec.emit, rec)
    assert rec.questions[0].options == ('Memory',)
    assert 'Where should generated records be stored?' in rec.questions[0].prompt
    assert fake.calls == [(f'/api/session/{SID}/form/frm_cancel', None)]


@pytest.mark.parametrize('busy', ['active', 'pending'])
def test_busy_session_rejected_before_selection_changes(fake, busy):
    if busy == 'active': fake.active = {SID: {'type': 'running'}}
    else: fake.pending = [{'id': 'msg_old'}]
    rec = Recorder()
    rt = Opencode2Runtime(ChatConfig(mode='plan'), base_url=fake.base_url)
    out = rt.run_turn(SESSION, SID, 'hi', 'openai/gpt', rec.emit, rec)
    assert out.status == 'failed'
    assert 'busy' in out.error
    assert fake.calls == []


@pytest.mark.parametrize('command,prompt,path', [('compact', '', 'compact'), ('', '/explain hello', 'command')])
def test_compact_and_slash_commands_wait_for_delivery(fake, command, prompt, path):
    rt, rec, result, worker = start_turn(fake, prompt=prompt, command=command)
    assert fake.calls[-1][0] == f'/api/session/{SID}/{path}'
    assert result == []
    fake.deliver()
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert result[0].status == 'completed'
    if path == 'command':
        assert fake.calls[0][1] == {'name': 'explain', 'text': 'hello'}


def test_files_are_inline_and_outside_mentions_are_excluded(tmp_path):
    (tmp_path / 'x.txt').write_text('hello')
    files = mention_files('@x.txt @../outside.txt', str(tmp_path))
    assert files == [{'uri': 'data:text/plain;base64,aGVsbG8=', 'name': 'x.txt',
                      'mention': {'start': 0, 'end': 6, 'text': '@x.txt'}}]


def test_models_use_location_wrapper(fake):
    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    assert rt.list_models(SESSION) == ['openai/gpt  GPT']


def test_typed_form_answers_and_conditions():
    rt = Opencode2Runtime(ChatConfig(), base_url='http://unused')
    replies = iter(['false', '2', '2.5', 'Red, Blue'])
    rec = SimpleNamespace(answer=lambda _: next(replies))
    form = {'fields': [{'key': 'bool', 'type': 'boolean'}, {'key': 'n', 'type': 'integer'},
                       {'key': 'num', 'type': 'number'},
                       {'key': 'colors', 'type': 'multiselect', 'options': [
                           {'label': 'Red', 'value': 'r'}, {'label': 'Blue', 'value': 'b'}]},
                       {'key': 'hidden', 'type': 'string', 'hidden': True, 'default': 'default'},
                       {'key': 'skipped', 'type': 'string', 'when': [{'key': 'bool', 'op': 'eq', 'value': True}]}]}
    assert rt._form_answer(form, rec) == {'bool': False, 'n': 2, 'num': 2.5,
                                         'colors': ['r', 'b'], 'hidden': 'default'}


def test_timeout_is_failure_not_hang(fake):
    rt, rec, result, worker = start_turn(fake)
    fake.deliver()
    finish(worker, rt)
    assert result[0].status == 'failed'
    assert 'timeout' in result[0].error
    assert any(isinstance(ev, Failure) for ev in rec.events)


def test_model_variant_is_a_ref_field(fake):
    rt, rec, result, worker = start_turn(fake, model='openai/gpt#medium')
    fake.deliver()
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert fake.calls[0] == (f'/api/session/{SID}/model',
                             {'model': {'providerID': 'openai', 'id': 'gpt', 'variant': 'medium'}})


@pytest.mark.parametrize('model', ['gpt', '/gpt', 'openai/', 'openai/gpt#', 'openai/gpt#a#b'])
def test_malformed_model_does_not_mutate_session(fake, model):
    rec = Recorder()
    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    assert rt.run_turn(SESSION, SID, 'hi', model, rec.emit, rec).status == 'failed'
    assert fake.calls == []


def test_plan_restores_after_admission_error(fake):
    fake.admission_error = True
    rec = Recorder()
    rt = Opencode2Runtime(ChatConfig(mode='plan'), base_url=fake.base_url)
    out = rt.run_turn(SESSION, SID, 'hi', '', rec.emit, rec)
    assert out.status == 'failed'
    assert 'bad prompt' in out.error
    assert fake.calls[-1] == (f'/api/session/{SID}/agent', {'agent': 'build'})
    rt.close()


def test_plan_without_prior_agent_fails_before_mutation(fake):
    fake.agent = None
    rec = Recorder()
    rt = Opencode2Runtime(ChatConfig(mode='plan'), base_url=fake.base_url)
    out = rt.run_turn(SESSION, SID, 'hi', 'openai/gpt', rec.emit, rec)
    assert out.status == 'failed'
    assert 'prior agent' in out.error
    assert fake.calls == []
    rt.close()


def test_same_runtime_rejects_overlapping_turn_before_http(fake):
    rt, rec, result, worker = start_turn(fake)
    other = Recorder()
    out = rt.run_turn(SESSION, SID, 'overlap', '', other.emit, other)
    assert out.status == 'failed'
    assert len(fake.calls) == 1
    fake.deliver()
    fake.publish('session.execution.succeeded')
    finish(worker, rt)


def test_usage_sums_distinct_steps_without_repeated_event_double_count():
    rt = Opencode2Runtime(ChatConfig(), base_url='http://unused')
    st, rec = TurnState(SID, delivered=True), Recorder()
    for message, inp, out in [('a', 10, 2), ('b', 12, 3), ('b', 12, 3)]:
        rt.handle_event({'type': 'session.step.ended', 'data': {
            'sessionID': SID, 'assistantMessageID': message,
            'tokens': {'input': inp, 'output': out}}}, st, rec.emit, rec)
    assert st.tokens == {'input': 22, 'output': 5}


def test_foreground_child_receives_private_password_in_environment(monkeypatch, tmp_path):
    calls = []
    proc = SimpleNamespace(stderr=io.StringIO(''), poll=lambda: None)

    def spawn(argv, **kwargs):
        calls.append((argv, kwargs))
        return proc

    monkeypatch.setattr('tandem.chat.runtime.opencode2.subprocess.Popen', spawn)
    rt = Opencode2Runtime(ChatConfig(), binary=['opencode'])
    monkeypatch.setattr(rt, '_healthy', lambda: True)
    rt.ensure_server(str(tmp_path), tandem_id='tdm-test')
    argv, kwargs = calls[0]
    assert argv[:2] == ['opencode', 'serve']
    assert '--service' not in argv
    assert kwargs['env']['OPENCODE_PASSWORD'] == rt.password
    assert rt.password not in argv
    assert kwargs['cwd'] == str(tmp_path)
    assert kwargs['start_new_session'] is True
    assert len(rt.password) >= 32
    rt._proc = None


def test_cancel_queued_input_without_a_running_execution(fake, monkeypatch):
    rt, rec, result, worker = start_turn(fake)
    original = rt._http

    def interrupt_no_active(method, path, body=None, timeout=30):
        if '/interrupt?' in path:
            fake.calls.append((path, body))
            return {'interrupted': False}
        return original(method, path, body, timeout)

    monkeypatch.setattr(rt, '_http', interrupt_no_active)
    own_id = fake.calls[0][1]['id']
    rt.interrupt()
    finish(worker, rt)
    assert result[0].status == 'interrupted'
    assert (f'/api/session/{SID}/inbox/{own_id}', None) in fake.calls


def test_progress_output_snapshots_do_not_repeat_at_tool_completion():
    rt = Opencode2Runtime(ChatConfig(), base_url='http://unused')
    st, rec = TurnState(SID, delivered=True), Recorder()
    for output in ('one', 'one\ntwo', 'one\ntwo'):
        rt.handle_event({'type': 'session.tool.progress', 'data': {
            'sessionID': SID, 'id': 'tool_x', 'metadata': {'output': output}}}, st, rec.emit, rec)
    rt.handle_event({'type': 'session.tool.success', 'data': {
        'sessionID': SID, 'id': 'tool_x', 'content': [{'type': 'text', 'text': 'one\ntwo\nthree'}]}},
        st, rec.emit, rec)
    assert rec.events == [ToolOutput('tool_x', 'one'), ToolOutput('tool_x', '\ntwo'),
                          ToolOutput('tool_x', '\nthree'), ToolFinished('tool_x', True)]


def test_tool_environment_excludes_server_passwords_and_keeps_session_context(fake, monkeypatch):
    monkeypatch.setenv('OPENCODE_PASSWORD', 'inherited-private-password')
    monkeypatch.setenv('OPENCODE_SERVER_PASSWORD', 'legacy-private-password')
    monkeypatch.setenv('CLAUDECODE', 'parent-session-marker')
    monkeypatch.setenv('TANDEM_TOOL_ENV_TEST', 'intended-tool-value')
    rt, rec, result, worker = start_turn(fake, ChatConfig(mode='plan'), model='openai/gpt')
    fake.deliver()
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert result[0].status == 'completed'
    assert len(fake.environments) == 1
    path, body, prior_posts = fake.environments[0]
    assert path == f'/api/session/{SID}/environment'
    assert prior_posts == 0  # registration precedes plan/model selection and admission
    variables = body['variables']
    assert 'OPENCODE_PASSWORD' not in variables
    assert 'OPENCODE_SERVER_PASSWORD' not in variables
    assert 'CLAUDECODE' not in variables
    assert variables['TANDEM_TOOL_ENV_TEST'] == 'intended-tool-value'
    from tandem.constants import SESSION_ENV
    assert variables[SESSION_ENV] == SESSION.tandem_id


@pytest.mark.parametrize('command,prompt', [('', 'hi'), ('compact', ''), ('', '/explain hello')])
def test_environment_registration_failure_prevents_admission_and_selection(fake, command, prompt):
    fake.environment_error = True
    rt = Opencode2Runtime(ChatConfig(mode='plan'), base_url=fake.base_url)
    rec = Recorder()
    out = rt.run_turn(SESSION, SID, prompt, 'openai/gpt', rec.emit, rec, command)
    assert out.status == 'failed'
    assert 'environment rejected' in out.error
    assert fake.calls == []
    assert len(fake.environments) == 1
    assert rec.events[-1] == TurnFinished('failed')
    rt.close()


def test_plan_restoration_settles_native_control_work_before_next_turn(fake):
    fake.restore_delay = .15
    fake.restore_reminder = PLAN_REMINDER
    rt, rec, result, worker = start_turn(fake, ChatConfig(mode='plan'))
    fake.deliver()
    fake.publish('session.execution.succeeded')
    worker.join(3)
    assert not worker.is_alive()
    assert result[0].status == 'completed'
    assert fake.restore_settled.is_set()
    assert fake.active == {} and fake.pending == [PLAN_REMINDER]
    assert rec.events[-1] == TurnFinished('completed')

    # Reuse the same runtime immediately, as the dispatcher does after return.
    rt.cfg = ChatConfig()
    fake.posted.clear()
    next_result = []
    next_worker = threading.Thread(target=lambda: next_result.append(
        rt.run_turn(SESSION, SID, 'next turn', '', rec.emit, rec)))
    next_worker.start()
    assert fake.posted.wait(2)
    path, body = next(c for c in reversed(fake.calls) if c[0].endswith('/prompt'))
    fake.publish('session.inbox.delivered', inboxID=body['id'])
    fake.publish('session.execution.succeeded')
    finish(next_worker, rt)
    assert next_result[0].status == 'completed'


@pytest.mark.parametrize('typ,delivery', [('user', 'steer'), ('user', 'queue'),
                                        ('synthetic', 'queue'), ('compaction', 'steer'),
                                        ('compaction', 'queue'), ('move', 'steer')])
def test_passive_reminder_does_not_hide_pending_executable_work(fake, typ, delivery):
    fake.pending = [PLAN_REMINDER, {'id': 'msg_other', 'type': typ, 'delivery': delivery}]
    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    assert rt._busy(SID) is True
    assert fake.pending[0] == PLAN_REMINDER


def test_passive_reminder_does_not_hide_active_execution(fake):
    fake.pending = [PLAN_REMINDER]
    fake.active = {SID: {'type': 'running'}}
    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    assert rt._busy(SID) is True


def test_inactive_reminder_survives_busy_and_idle_checks(fake):
    fake.pending = [PLAN_REMINDER]
    rt = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    assert rt._busy(SID) is False
    rt._await_idle(SID, timeout=.1)
    assert fake.pending == [PLAN_REMINDER]
    assert fake.calls == []


def test_plan_slash_command_ignores_reminder_before_user_admission(fake):
    rt, rec, result, worker = start_turn(fake, ChatConfig(mode='plan'), prompt='/explain hello')
    reminder_item = {key: PLAN_REMINDER[key] for key in ('type', 'delivery', 'payload')}
    fake.publish('session.inbox.enqueued', inboxID=PLAN_REMINDER['id'], item=reminder_item)
    fake.publish('session.inbox.delivered', inboxID=PLAN_REMINDER['id'])
    fake.publish('session.execution.succeeded')
    time.sleep(.05)
    assert result == []
    assert rec.events == []
    fake.deliver()
    fake.publish('session.text.delta', assistantMessageID='msg_a', ordinal=0, delta='Command response')
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert result[0].status == 'completed'
    assert rec.events == [TextDelta('Command response'), TurnFinished('completed')]
    assert fake.calls[-1] == (f'/api/session/{SID}/agent', {'agent': 'build'})


def test_slash_callback_without_user_admission_does_not_claim_completion(fake):
    rt, rec, result, worker = start_turn(fake, prompt='/explain hello')
    fake.publish('session.inbox.enqueued', inboxID=PLAN_REMINDER['id'], item={
        key: PLAN_REMINDER[key] for key in ('type', 'delivery', 'payload')})
    fake.publish('session.inbox.delivered', inboxID=PLAN_REMINDER['id'])
    fake.publish('session.execution.succeeded')
    finish(worker, rt)
    assert result[0].status == 'failed'
    assert 'timeout' in result[0].error
    assert rec.events[-1] == TurnFinished('failed')


def test_unknown_command_event_shape_is_not_guessed_as_admission():
    rt = Opencode2Runtime(ChatConfig(), base_url='http://unused')
    st, rec = TurnState(SID), Recorder()
    rt.handle_event({'type': 'session.inbox.enqueued', 'data': {
        'sessionID': SID, 'inboxID': 'msg_untyped'}}, st, rec.emit, rec)
    assert st.inbox_id == ''
    assert st.delivered is False


@pytest.mark.parametrize('phase', [None, 'sessions', 'completed'])
@pytest.mark.parametrize('entry', ['turn', 'models'])
def test_retained_legacy_history_rejected_before_local_server_start(tmp_path, monkeypatch, phase, entry):
    import sqlite3
    from tandem.harness import opencode2 as storage

    db = tmp_path / 'legacy.db'
    conn = sqlite3.connect(db)
    try:
        conn.executescript("CREATE TABLE session (id TEXT PRIMARY KEY); CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT);")
        conn.execute('INSERT INTO session VALUES (?)', (SID,))
        if phase is not None:
            conn.execute('INSERT INTO kv VALUES (?, ?)', ('migration.v1-v2', json.dumps({'phase': phase})))
        conn.commit()
    finally:
        conn.close()
    before = db.read_bytes()
    monkeypatch.setattr(storage, 'db_path', lambda: db)
    rt = Opencode2Runtime(ChatConfig())
    monkeypatch.setattr(rt, 'ensure_server', lambda *args, **kwargs: pytest.fail('legacy history must not start a native server'))
    rec = Recorder()
    try:
        if entry == 'turn':
            out = rt.run_turn(SESSION, SID, 'hi', '', rec.emit, rec)
            assert out.status == 'failed'
            assert 'Retained OpenCode 1 session' in out.error
            assert rec.events[-1] == TurnFinished('failed')
        else:
            session = SimpleNamespace(cwd=SESSION.cwd, tandem_id=SESSION.tandem_id, native_id=lambda _: SID)
            with pytest.raises(storage.LegacySessionUnsupported, match='Retained OpenCode 1 session'):
                rt.list_models(session)
        assert db.read_bytes() == before
    finally:
        rt.close()


def test_local_runtime_requires_a_discoverable_database_before_server_start(monkeypatch):
    from tandem.harness import opencode2 as storage

    monkeypatch.setattr(storage, 'db_path', lambda: None)
    rt = Opencode2Runtime(ChatConfig())
    monkeypatch.setattr(rt, 'ensure_server', lambda *args, **kwargs: pytest.fail('unknown history must not start a native server'))
    rec = Recorder()
    try:
        out = rt.run_turn(SESSION, SID, 'hi', '', rec.emit, rec)
        assert out.status == 'failed'
        assert 'database is not discoverable' in out.error
        assert rec.events[-1] == TurnFinished('failed')
    finally:
        rt.close()


def test_fresh_local_runtime_identity_is_allowed_beside_pending_legacy_migration(tmp_path, monkeypatch):
    import sqlite3
    from tandem.harness import opencode2 as storage

    db = tmp_path / 'mixed.db'
    conn = sqlite3.connect(db)
    try:
        conn.executescript("CREATE TABLE session (id TEXT PRIMARY KEY); INSERT INTO session VALUES ('ses_old');")
    finally:
        conn.close()
    before = db.read_bytes()
    monkeypatch.setattr(storage, 'db_path', lambda: db)
    rt = Opencode2Runtime(ChatConfig())
    try:
        rt._check_local_history(SID)
        assert db.read_bytes() == before
    finally:
        rt.close()


@pytest.mark.parametrize('status,tag', [(409, 'FormAlreadySettledError'), (404, 'FormNotFoundError')])
@pytest.mark.parametrize('key', [b'\x1b', b'\x03'])
def test_window_cancel_when_interrupt_already_settled_form(fake, env_factory, monkeypatch, status, tag, key):
    from test_chat_window import make_window

    env = env_factory()
    w, dispatcher, _, answers = make_window(env)
    posted = queue.Queue()
    monkeypatch.setattr(answers, '_post', posted.put)
    runtime = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    runtime._session_id = SID
    monkeypatch.setattr(dispatcher, 'interrupt', runtime.interrupt)
    fake.form_cancel_error = (status, {'_tag': tag, 'id': 'frm_race', 'message': 'settled'})
    fake.form_cancel_after_interrupt = True
    failures = []
    form = {'id': 'frm_race', 'sessionID': SID, 'fields': [{'key': 'question', 'type': 'string'}]}

    def handle():
        try:
            runtime.handle_event({'type': 'form.created', 'data': {'form': form}},
                                 TurnState(SID, delivered=True), w.handle_event, answers)
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=handle, daemon=True)
    try:
        worker.start()
        w.handle_event(posted.get(timeout=2))
        dispatcher.busy = True
        w.handle_input(key)
        worker.join(3)
        assert not worker.is_alive()
        assert failures == []
        assert fake.interrupted.is_set()
        assert sorted(fake.calls) == sorted([
            (f'/api/session/{SID}/form/frm_race', None),
            (f'/api/session/{SID}/interrupt?resume=false', None),
        ])
        assert w.composer.mode == 'prompt'
    finally:
        answers.close()
        worker.join(3)
        env.store.close()


@pytest.mark.parametrize('status,error', [
    (401, {'id': 'frm_error'}),
    (403, {'_tag': 'FormAlreadySettledError', 'id': 'frm_error'}),
    (500, {'_tag': 'FormAlreadySettledError', 'id': 'frm_error'}),
    (409, {'_tag': 'ConflictError', 'id': 'frm_error'}),
    (404, {'_tag': 'SessionNotFoundError', 'id': SID}),
    (409, {'_tag': 'FormAlreadySettledError', 'id': 'another-form'}),
    (409, {'id': 'frm_error'}),
])
def test_question_cancel_preserves_unrelated_delete_errors(fake, status, error):
    from tandem.chat.runtime.opencode2 import Opencode2HttpError
    from tandem.chat.window import WindowAnswers

    fake.form_cancel_error = (status, error)
    answers = WindowAnswers(lambda req: answers.cancel_question())
    runtime = Opencode2Runtime(ChatConfig(), base_url=fake.base_url)
    form = {'id': 'frm_error', 'sessionID': SID, 'fields': [{'key': 'question', 'type': 'string'}]}
    with pytest.raises(Opencode2HttpError) as raised:
        runtime.handle_event({'type': 'form.created', 'data': {'form': form}},
                             TurnState(SID, delivered=True), lambda ev: None, answers)
    assert raised.value.status == status
    assert raised.value.data == error
    assert fake.calls == [(f'/api/session/{SID}/form/frm_error', None)]


def test_question_cancel_preserves_delete_connection_failure(monkeypatch):
    from tandem.chat.window import WindowAnswers

    answers = WindowAnswers(lambda req: answers.cancel_question())
    runtime = Opencode2Runtime(ChatConfig(), base_url='http://127.0.0.1:1')
    calls = []

    def disconnected(method, path, body=None):
        calls.append((method, path, body))
        raise ConnectionError('cancel connection lost')

    monkeypatch.setattr(runtime, '_http', disconnected)
    form = {'id': 'frm_error', 'sessionID': SID, 'fields': [{'key': 'question', 'type': 'string'}]}
    with pytest.raises(ConnectionError, match='cancel connection lost'):
        runtime.handle_event({'type': 'form.created', 'data': {'form': form}},
                             TurnState(SID, delivered=True), lambda ev: None, answers)
    assert calls == [('DELETE', f'/api/session/{SID}/form/frm_error', None)]
