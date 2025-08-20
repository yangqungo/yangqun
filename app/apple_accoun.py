import os
import re
import json
import re
import json
import requests
from typing import Any, Dict, List, Union

JSONType = Union[Dict[str, Any], List[Any]]





def parse_http_request(request_text):
    """
    解析 HTTP 请求文本，提取键值对并返回格式化后的 JSON 对象。
    :param request_text: HTTP 请求文本（curl 或 HTTP 请求格式）
    :return: 格式化后的 JSON 对象
    """
    result = {}
    # 解析 User-Agent
    user_agent_match = re.search(r"(?:User-Agent: |-H 'User-Agent: )(.+?)(?:\r\n|\n|'|$)", request_text)
    if user_agent_match:
        result["User-Agent"] = user_agent_match.group(1)
    
    # 解析 playlet.api_st
    playlet_match = re.search(r"playlet.api_st=([^;]+)", request_text)
    if playlet_match:
        result["playlet.api_st"] = playlet_match.group(1)
    
    # 解析 userId
    user_id_match = re.search(r"userId=([^;]+)", request_text)
    if user_id_match:
        result["userId"] = user_id_match.group(1)
    
    # 解析 passToken
    pass_token_match = re.search(r"passToken=([^;]+)", request_text)
    if pass_token_match:
        result["passToken"] = pass_token_match.group(1)
    
    # 解析 did
    did_match = re.search(r"did=([^;]+)", request_text)
    if did_match:
        result["did"] = did_match.group(1)
    
    # 解析 message
    message_match = re.search(r'"message":"([^"]+)"', request_text)
    if message_match:
        result["message"] = message_match.group(1)
    
    result.update({
                     "proxies":{'http': '', 'https': ''}
                     
                     })
    
    return result

def read_and_parse_file(file_path,token, url):
    """
    读取文件并根据内容解析为字典或调用 parse_http_request
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
            # 判断是否为JSON格式
            if content.startswith('{') or content.startswith('['):
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON format in file {file_path}: {e}")
            else:
                print(f"Non-JSON content in file {file_path}, parsing as HTTP request")
                parsed_dict = parse_http_request(content)
                
                
                message = parsed_dict['message'].encode().decode('unicode_escape').replace('\r\n', '').replace('\/', '/')
                
                data = {"operation":"decrypt","data":message}
                headers = {"Content-Type": "application/json","X-Token":token}
                finally_url = url + '/process'
                result = requests.post(finally_url, headers=headers, data=json.dumps(data)).json()["result"]

                
                decrypt_messag = json.loads(result)

                extract_messag = extract_info(decrypt_messag)

                merged = parsed_dict | extract_messag

                
            
                if merged:
                    try:
                        print("格式化账号数据")
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(json.dumps(merged, indent=4) + '\n')
                    except Exception as write_error:
                        print(f"Failed to write to file {file_path}: {write_error}")
                return merged
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Permission denied when reading file: {file_path}")
        return None
    except Exception as e:
        print(f"Unexpected error reading or parsing file {file_path}: {e}")
        return None

def process_xf_files(directory,token, url):
    """
    处理同目录下以 'xf_' 开头的 .text 或 .txt 文件
    """
    for filename in os.listdir(directory):
        if filename.startswith('xf_') and (filename.endswith('.text') or filename.endswith('.txt') or '.' not in filename):
            file_path = os.path.join(directory, filename)
            read_and_parse_file(file_path,token, url)
            # return result 




def start_account(token, url):
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    process_xf_files(base_dir,token, url)


def extract_token_and_url():
    directory = os.path.dirname(os.path.abspath(__file__))

    token_xf = None
    url_xf = None

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")

    for filename in os.listdir(directory):
        if filename.lower().startswith("token_xf") and filename.lower().endswith((".text", ".txt")):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError(f"文件 '{filename}' 为空")
                token_xf = content
            except Exception as e:
                raise ValueError(f"解析 token_xf 文件失败: {e}")

        elif filename.lower().startswith("url_xf") and filename.lower().endswith((".text", ".txt")):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError(f"文件 '{filename}' 为空")
                url_xf = content
            except Exception as e:
                raise ValueError(f"解析 url_xf 文件失败: {e}")

    if token_xf is None:
        raise ValueError("未找到 token_xf 文件")
    if url_xf is None:
        raise ValueError("未找到 url_xf 文件")
    
    

    url = f"{url_xf.rstrip('/')}/token_remaining"
    
    params = {
            "token": token_xf,
            "collection": "tokens",   # 或者 "ks_token"
        }
    resp = requests.get(url, params=params, timeout=(5, 20))
    data = resp.json()
    print(f"token剩余次数:{data['remaining']}")

    return token_xf, url_xf


def extract_info(data: dict) -> dict:
    """
    从上报 JSON 提取关键信息（递归查找）
    自动遍历嵌套的 dict / list，返回第一个匹配的值
    """
    JSONType = Union[Dict[str, Any], List[Any]]

    def normalize_key(key: str) -> str:
        """忽略大小写、下划线、短横线"""
        return ''.join(ch for ch in key.lower() if ch.isalnum())

    def find_first(d: JSONType, target_key: str) -> Any:
        """递归在 d 中查找第一个匹配 target_key 的值"""
        norm_target = normalize_key(target_key)

        if isinstance(d, dict):
            for k, v in d.items():
                if normalize_key(k) == norm_target:
                    return v
            for v in d.values():
                if isinstance(v, (dict, list)):
                    result = find_first(v, target_key)
                    if result is not None:
                        return result

        elif isinstance(d, list):
            for item in d:
                if isinstance(item, (dict, list)):
                    result = find_first(item, target_key)
                    if result is not None:
                        return result

        return None

    fields = [
        "appId",
        "egid",
        "deviceId",
        "kenyIdList",
        "systemBootTime",
        "imsi",
        "wifiList",
        "idfv",
        "networkInfo",
        "deviceFileTime",
        "systemUpdateTime"
                
        
    ]
    re =  {field: find_first(data, field) for field in fields}
    if isinstance(re.get("wifiList"), list) and not re["wifiList"]: 
        re["wifiList"] = [
                {
                    "bssid": "24:69:67:fd:fd:a2",
                    "ssid": "kei"
                }
            ]

    return re
token, url = extract_token_and_url()
start_account(token, url)
