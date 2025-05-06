from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen1.5-0.5b-chat-hf',
        path='/root/.cache/modelscope/hub/models/Qwen/Qwen1___5-0___5B-Chat', # 这里无论用那种方式都要写死
        max_out_len=1024,
        batch_size=32,
        run_cfg=dict(num_gpus=1),
        stop_words=['<|im_end|>', '<|im_start|>'],
    )
]
