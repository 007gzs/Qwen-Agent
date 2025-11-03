

class ChatApi:

    def __init__(self, agent):
        self.agent = agent

    def run_webui(self, chatbot_config=None, server_name='0.0.0.0', server_port=7860):
        from qwen_agent.gui import WebUI
        WebUI(
            self.agent,
            chatbot_config=chatbot_config,
        ).run(server_name=server_name, server_port=server_port)

    def run_apiserver(self, server_name='0.0.0.0', server_port=8080):
        from .api_server import start_apiserver
        start_apiserver(self, server_name, server_port)

    def get_type(self, item):
        chunk = 'chunk' in item
        if 'function_call' in item:
            return 'function_call'
        elif item['role'] == 'function':
            return 'function_call_output'
        elif 'reasoning_content' in item and item['reasoning_content']:
            if chunk:
                return 'reasoning_chunk'
            else:
                return 'reasoning'
        else:
            if chunk:
                return 'message_chunk'
            else:
                return 'message'

    def add_type(self, item):
        item['type'] = self.get_type(item)
        return item

    def gen_stream(self, response, add_full_msg=False):
        last = []
        last_msg = ""
        for rsp in response:
            now = rsp[-1]
            if last and now == last[-1]:
                last = rsp
                continue
            now_type = self.get_type(now)
            is_new_line = len(last) != len(rsp)
            if is_new_line and last:
                res = self.add_type(last[-1])
                last_msg = ''
                if add_full_msg or res['type'] not in ('message', 'reasoning'):
                    yield res
            content_key = {
                "message": "content",
                "reasoning": "reasoning_content"
            }
            if now_type in ('message', 'reasoning'):
                msg = now[content_key[now_type]]
                assert msg.startswith(last_msg)
                stream_msg = msg[len(last_msg):]
                yield {'role': now['role'], 'content': '', 'reasoning_content': '', 'chunk': stream_msg, 'type': now_type}
                last_msg = msg
            last = rsp
            
        if last:
            res = self.add_type(last[-1])
            last_msg = ''
            if add_full_msg or res['type'] not in ('message', 'reasoning'):
                yield res

    def gen(self, response):
        last = None
        for rsp in response:
            if last is not None:
                if len(last) != len(rsp):
                    yield last[-1]
            last = rsp
        if last:
            yield last[-1]

    def chat(self, messages, stream=True, add_full_msg=False, **kwargs):
        response = self.agent.run(messages, **kwargs)
        if stream:
            return self.gen_stream(response, add_full_msg=add_full_msg)
        else:
            return self.gen(response)
