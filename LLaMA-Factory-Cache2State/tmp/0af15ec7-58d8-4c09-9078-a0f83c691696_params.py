datasets = [
    [
        dict(
            abbr='ruler_niah_multiquery_8k',
            base_path='./data/ruler',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.datasets.ruler.ruler_niah.RulerNiahEvaluator')
            ),
            file_path='PaulGrahamEssays.jsonl',
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(prompt='{prompt}', role='HUMAN'),
                        dict(prompt='{answer}\n', role='BOT'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            max_seq_length=8192,
            num_needle_k=1,
            num_needle_q=4,
            num_needle_v=1,
            num_samples=100,
            reader_cfg=dict(
                input_columns=[
                    'prompt',
                ], output_column='answer'),
            tokenizer_model='gpt-4',
            tokens_to_generate=128,
            type='opencompass.datasets.ruler.ruler_niah.RulerNiahDataset',
            type_haystack='essay',
            type_needle_k='words',
            type_needle_v='numbers'),
    ],
]
eval = dict(runner=dict(task=dict()))
models = [
    dict(
        abbr='l4-g512-[0]',
        batch_size=1,
        max_out_len=50,
        max_seq_len=32768,
        model_kwargs=dict(
            attn_implementation='flash_attention_2',
            device_map='cuda',
            replaced_layers=[
                0,
            ],
            torch_dtype='torch.bfloat16',
            trust_remote_code=True),
        path=
        '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/models/Llama-3.2-3B-fla-hybrid-sort2-global-local',
        run_cfg=dict(num_gpus=1, num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path=
        '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/models/Llama-3.2-3B-fla-hybrid-sort2-global-local',
        type=
        'opencompass.models.myModel.hf_strip_model.HuggingFaceCausalLM_Strip'),
]
work_dir = '/inspire/hdd/ws-8207e9e2-e733-4eec-a475-cfa1c36480ba/embodied-multimodality/liuxiaoran-240108120089/train/opencompass_eval_result/local4-global512/20250417_181925'
