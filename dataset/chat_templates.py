CHATML_TEMPLATE =  "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


LLAMA3_TEMPLATE = """"{{- bos_token }}
{%- if not add_generation_prompt is defined %}
    {%- set add_generation_prompt = false %}
{%- endif %}

{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content']|trim + '<|eot_id|>' }}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


GEMMA_2 = """{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}
{%- if messages[0]['role'] != 'system' -%}
    {%- for message in messages -%}
        {%- if message['role'] == 'assistant' -%}
            {%- set role = 'model' -%}
        {%- else -%}
            {%- set role = message['role'] -%}
        {%- endif -%}
        {{- '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' -}}
    {%- endfor -%}
    {%- if add_generation_prompt -%}
        {{- '<start_of_turn>model\n' -}}
    {%- endif -%}
{%- endif -%}
"""