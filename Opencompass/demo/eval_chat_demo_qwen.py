from mmengine.config import read_base



with read_base():
    # 数据读入
    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.ceval.ceval_gen_5f30c7 import ceval_datasets  # noqa: F401, F403
    # from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets  # noqa: F401, F403
    from opencompass.configs.datasets.demo.demo_gsm8k_base_gen import gsm8k_datasets

    # 模型读入
    from opencompass.configs.models.qwen.hf_qwen1_5_0_5b_chat import \
        models as hf_qwen1_5_0_5b_chat_1_5b_models

    from opencompass.configs.models.qwen2_5.hf_qwen2_5_1_5b_instruct import \
        models as hf_qwen2_5_1_5b_instruct_1_5b_models

datasets = gsm8k_datasets + cmmlu_datasets + ceval_datasets
models = hf_qwen1_5_0_5b_chat_1_5b_models + hf_qwen2_5_1_5b_instruct_1_5b_models
# models = hf_qwen1_5_0_5b_chat_1_5b_models + hf_internlm2_chat_1_8b_models
