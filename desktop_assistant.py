import json
import sqlite3
import os
from datetime import datetime

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print, multimodal_typewriter_print

# SQLite database initialisation
DB_PATH = "desktop_history.db"

def init_db():
    """Create the history table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS desktop_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            objects TEXT NOT NULL,          -- JSON string with detected objects
            description TEXT NOT NULL       -- Natural-language description
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Record observation into SQLite
@register_tool('record_desktop_history')
class RecordDesktopHistory(BaseTool):
    description = (
        "Record the current observation of the desktop. "
        "Call this tool after analysing an image to store which objects "
        "(cups, boxes, pens, etc.) are visible and where they are located."
    )
    parameters = {
        'type': 'object',
        'properties': {
            'image_path': {
                'type': 'string',
                'description': 'Path to the image that was analysed.'
            },
            'objects': {
                'type': 'string',
                'description': (
                    'A JSON string containing a list of objects and their '
                    'locations, e.g. [{"name": "cup", "location": "left side"}, '
                    '{"name": "pen", "location": "center"}].'
                )
            },
            'description': {
                'type': 'string',
                'description': 'A short natural-language description of the scene.'
            }
        },
        'required': ['image_path', 'objects', 'description']
    }

    def call(self, params: str, **kwargs) -> str:
        # Parse and validate the JSON arguments
        params_dict = self._verify_json_format_args(params)
        image_path = params_dict['image_path']
        objects = params_dict['objects']
        description = params_dict['description']
        timestamp = datetime.now().isoformat(timespec='seconds')

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO desktop_history (timestamp, image_path, objects, description) "
            "VALUES (?, ?, ?, ?)",
            (timestamp, image_path, objects, description)
        )
        conn.commit()
        conn.close()

        return json.dumps({
            'status': 'ok',
            'message': f'History recorded at {timestamp}.'
        }, ensure_ascii=False)

# 4. Custom tool: query historical observations from SQLite
@register_tool('query_desktop_history')
class QueryDesktopHistory(BaseTool):
    description = (
        "Query past observations of the desktop. "
        "Use this tool to answer questions about where objects were located earlier, "
        "whether a new photo changed a previous judgement, or to list all records "
        "related to a specific object (e.g. cups, boxes, pens)."
    )
    parameters = {
        'type': 'object',
        'properties': {
            'object_name': {
                'type': 'string',
                'description': (
                    'Optional. Filter records that contain this object name '
                    '(e.g. "cup", "box", "pen"). If omitted, return all records.'
                )
            },
            'limit': {
                'type': 'integer',
                'description': 'Maximum number of records to return (default: 10).'
            }
        },
        'required': []   # no required parameters
    }

    def call(self, params: str, **kwargs) -> str:
        params_dict = self._verify_json_format_args(params)
        object_name = params_dict.get('object_name')
        limit = params_dict.get('limit', 10)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if object_name:
            # Simple LIKE filter on the JSON objects column
            cursor.execute(
                "SELECT timestamp, image_path, objects, description "
                "FROM desktop_history "
                "WHERE objects LIKE ? "
                "ORDER BY id DESC LIMIT ?",
                (f'%{object_name}%', limit)
            )
        else:
            cursor.execute(
                "SELECT timestamp, image_path, objects, description "
                "FROM desktop_history "
                "ORDER BY id DESC LIMIT ?",
                (limit,)
            )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return json.dumps({
                'status': 'empty',
                'message': 'No matching history records found.'
            }, ensure_ascii=False)

        records = []
        for row in rows:
            records.append({
                'timestamp': row[0],
                'image_path': row[1],
                'objects': json.loads(row[2]),
                'description': row[3]
            })

        return json.dumps({
            'status': 'ok',
            'count': len(records),
            'records': records
        }, ensure_ascii=False)


# LLM configuration
llm_cfg = {
    'model': 'qwen3-vl-plus',
    'model_type': 'qwenvl_dashscope',
    # 'api_key': 'YOUR_DASHSCOPE_API_KEY',   
    'generate_cfg': {
        'top_p': 0.8,
        'top_k': 20,
        'temperature': 0.7,
        'repetition_penalty': 1.0,
        'presence_penalty': 1.5,
    }
}

analysis_prompt = """
You are a desktop observation assistant.
When the user provides an image, follow this iterative loop:
1. Look closely at the image.
2. Use the provided tools to find information (zoom in, record history, query history).
3. Review your findings and decide on the next action.
Repeat until you can give a synthesised answer.
Always record what you see with the record_desktop_history tool after each analysis.
When asked about past locations or whether a new photo changed a previous judgement,
use the query_desktop_history tool to retrieve the relevant records.
"""

tools = ['image_zoom_in_tool',
         'record_desktop_history',
         'query_desktop_history']
agent = Assistant(
    llm=llm_cfg,
    function_list=tools,
    system_message=analysis_prompt,
)

messages = []
user_msg = {
    'role': 'user',
    'content': [
        {'text': 'What is in the photo? Please describe the objects and their locations.'},
        {'image': './examples/resource/desktop1.jpg'}
    ]
}
messages.append(user_msg)

response_plain_text = ''
for response in agent.run(messages=messages):
    response_plain_text = multimodal_typewriter_print(response, response_plain_text)
messages.extend(response)

messages = []
user_msg = {
    'role': 'user',
    'content': [
        {'text': 'Has a new photo changed your previous judgement? Please compare the two photos and describe any differences.'},
        {'image': './examples/resource/desktop2.jpg'}
    ]
}
messages.append(user_msg)

response_plain_text = ''
for response in agent.run(messages=messages):
    response_plain_text = multimodal_typewriter_print(response, response_plain_text)
messages.extend(response)