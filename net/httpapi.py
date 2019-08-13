import requests
import json
import time
import socket
import uuid


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0, 11, 2)])


class RequestsWrapper():
    def __init__(self, url, cookies_key=None):
        self.headers = {"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:51.0) Gecko/20100101 Firefox/51.0"}
        self.url = url
        self.cookies_key = cookies_key
        self.cookies = None
        if self.cookies_key:
            url_tmp = url + '/api/authenticate'
            post_params_tmp = {
                'email': '865295386@qq.com',
                'passwd': '9a03c36440f6161622a45734fed0409e23da660f'
            }
            r_tmp = requests.post(url_tmp, json=post_params_tmp, headers=self.headers)
            self.cookies = r_tmp.cookies

    def get(self, api):
        try:
            r = requests.get(self.url + api, headers=self.headers, cookies=self.cookies)
            r.encoding = 'UTF-8'
            json_response = json.loads(r.text)
            return json_response
        except Exception as e:
            print('get请求出错,出错原因:%s'%e)
            return {}

    def post(self, api, post_params):
        try:
            r = requests.post(self.url + api, json=post_params, headers=self.headers, cookies=self.cookies)
            json_response = json.loads(r.text)
            return json_response
        except Exception as e:
            print('post请求出错,原因:%s' % e)

    def delfile(self, url, params):
        try:
            del_word = requests.delete(url, params, headers=self.headers)
            json_response = json.loads(del_word.text)
            return json_response
        except Exception as e:
            print('del请求出错,原因:%s' % e)
            return {}

    def putfile(self, url, params):
        try:
            data = json.dumps(params)
            me = requests.put(url, data)
            json_response = json.loads(me.text)
            return json_response
        except Exception as e:
            print('put请求出错,原因:%s'%e)
            return json_response


def get_local_api_online_devices():

    req = RequestsWrapper(url='http://127.0.0.1:9000', cookies_key='awesession')
    api = '/api/online_devices'
    try:
        print(req.get(api))
    except ValueError as e:
        print('get error: %s' % e)


def post_local_api_devices():


    req = RequestsWrapper(url='http://127.0.0.1:9000', cookies_key='awesession')

    myname = socket.getfqdn(socket.gethostname())
    myaddr = socket.gethostbyname(myname)
    mymac = get_mac_address()

    post_params = {
        'name': myname,
        'addr': myaddr,
        'mac': mymac
    }
    api = '/api/devices'
    while True:
        print(req.post(api, post_params))
        time.sleep(3)


if __name__ == '__main__':
    post_local_api_devices();
    # get_local_api_online_devices()
